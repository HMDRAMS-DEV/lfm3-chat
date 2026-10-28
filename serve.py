#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch", "torchvision", "torchcodec", "transformers>=5.20.2", "accelerate",
#   "pillow", "librosa", "soundfile", "numpy", "fastapi", "uvicorn",
# ]
# ///
"""Local camera and voice chat with LiquidAI/LFM3-Omni-2B.

Run after the model download finishes:
    uv run --python 3.12 serve.py
    uv run --python 3.12 serve.py --model /path/to/LFM3-Omni-2B

Then open http://localhost:8000 and allow camera/microphone access. Speech is
submitted after a pause, together with up to six camera frames from that turn.
The model returns text and a 24 kHz spoken reply. This is turn-based chat;
recording pauses during inference and playback. No separate ASR or TTS model.

Only cached model files are used by default. --download permits Hugging Face
downloads. Custom model code is loaded with trust_remote_code=True, as required
by this checkpoint. --ui-only serves the interface without loading weights.

Model API: https://huggingface.co/LiquidAI/LFM3-Omni-2B
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import logging
import tempfile
import threading
import time
import wave
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

LOG = logging.getLogger("lfm3")
INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
MAX_SECONDS = 30
MAX_FRAMES = 6
MAX_BODY = 5_000_000
SYSTEM_PROMPT = (
    "You are having a live video conversation. Listen to the user's speech and "
    "use the camera frames when relevant. Frames are in chronological order, "
    "with times relative to the start of the recording. If no frames are provided, "
    "you cannot see the user. Answer the user's spoken request directly. "
    "Keep replies brief and conversational. Respond with interleaved text and audio."
)


def decode_base64(value: object, limit: int) -> bytes:
    if not isinstance(value, str) or len(value) > (limit + 2) // 3 * 4:
        raise ValueError("Missing or oversized media data.")
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 media data.") from exc
    if len(data) > limit:
        raise ValueError("Media exceeds the size limit.")
    return data


def validate_turn(payload: object) -> tuple[str, bytes, list[tuple[float, bytes]]]:
    """Validate browser media before it reaches the model or session storage."""
    from PIL import Image, UnidentifiedImageError

    if not isinstance(payload, dict):
        raise TypeError("Expected a JSON object.")
    session_id = payload.get("session_id")
    if (not isinstance(session_id, str) or not 1 <= len(session_id) <= 64
            or not all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in session_id)):
        raise ValueError("Invalid session ID.")
    audio = decode_base64(payload.get("audio"), INPUT_RATE * 2 * MAX_SECONDS + 4096)
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, INPUT_RATE, "NONE"):
                raise ValueError("Audio must be mono 16 kHz, 16-bit PCM WAV.")
            samples = wav.getnframes()
            if not INPUT_RATE // 10 <= samples <= INPUT_RATE * MAX_SECONDS:
                raise ValueError(f"Record between 0.1 and {MAX_SECONDS} seconds.")
            if len(wav.readframes(samples)) != samples * 2:
                raise ValueError("The audio file is truncated.")
            duration = samples / INPUT_RATE
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid WAV recording.") from exc
    frames = payload.get("frames", [])
    if not isinstance(frames, list) or len(frames) > MAX_FRAMES:
        raise ValueError(f"Send at most {MAX_FRAMES} camera frames.")
    decoded = []
    last_time = -1.0
    for frame in frames:
        if not isinstance(frame, dict):
            raise TypeError("Invalid camera frame.")
        timestamp = frame.get("time")
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not 0 <= timestamp <= duration + 0.5 or timestamp < last_time):
            raise ValueError("Camera timestamps must be chronological and inside the recording.")
        data = decode_base64(frame.get("image"), 300_000)
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG" or max(image.size) > 640 or min(image.size) < 1:
                    raise ValueError("Camera frames must be JPEG images up to 640 pixels per side.")
                image.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Invalid camera image.") from exc
        decoded.append((float(timestamp), data))
        last_time = timestamp
    return session_id, audio, decoded


@dataclass
class Turn:
    directory: tempfile.TemporaryDirectory
    user: dict
    assistant: dict


@dataclass
class Session:
    turns: list[Turn] = field(default_factory=list)
    touched: float = field(default_factory=time.monotonic)

    def clear(self):
        for turn in self.turns:
            turn.directory.cleanup()
        self.turns.clear()


class Engine:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self.ready = False
        self.device = "unloaded"

    def load(self):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        device = self.args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        dtype = self.args.dtype
        if dtype == "auto":
            dtype = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device.startswith("cuda") else "float32"
        self.device = device
        kwargs = {"trust_remote_code": True, "local_files_only": not self.args.download}
        LOG.info("Loading %s on %s with %s", self.args.model, device, dtype)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.args.model, dtype=getattr(torch, dtype), device_map=device, **kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.args.model, **kwargs)
        # The codec stays in its own float32 dtype, as in Liquid AI's example.
        self.codec = self.processor.audio_tokenizer.to(device=device).eval()
        self.context_length = getattr(self.model.config, "max_position_embeddings", 32768)
        self.ready = True
        LOG.info("Model ready")

    def close(self):
        for session in self.sessions.values():
            session.clear()
        self.sessions.clear()

    def get_session(self, session_id: str) -> Session:
        now = time.monotonic()
        for key in list(self.sessions):
            if now - self.sessions[key].touched > 3600:
                self.sessions.pop(key).clear()
        session = self.sessions.pop(session_id, None) or Session()
        session.touched = now
        self.sessions[session_id] = session
        while len(self.sessions) > 8:
            self.sessions.popitem(last=False)[1].clear()
        return session

    def reset(self, session_id: str):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "Wait for the current reply before clearing the conversation.")
        try:
            session = self.sessions.pop(session_id, None)
            if session:
                session.clear()
        finally:
            self.lock.release()

    def respond(self, session_id: str, audio: bytes, frames: list) -> dict:
        if not self.ready:
            raise HTTPException(503, "UI-only mode. Restart without --ui-only to load the model.")
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "The model is answering another turn. Please try again shortly.")
        directory = None
        try:
            started = time.monotonic()
            session = self.get_session(session_id)
            directory = tempfile.TemporaryDirectory(prefix="lfm3-turn-")
            root = Path(directory.name)
            audio_path = root / "user.wav"
            audio_path.write_bytes(audio)
            content = []
            for index, (timestamp, image) in enumerate(frames):
                image_path = root / f"frame-{index}.jpg"
                image_path.write_bytes(image)
                content.extend([
                    {"type": "text", "text": f"Camera at {timestamp:.1f}s:"},
                    {"type": "image", "image": str(image_path)},
                ])
            content.append({"type": "audio", "audio": str(audio_path)})
            user = {"role": "user", "content": content}
            # Reprocess the original recordings so follow-ups retain what was
            # actually heard and seen. Drop old turns before exceeding context.
            while True:
                conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
                for turn in session.turns:
                    conversation.extend([turn.user, turn.assistant])
                conversation.append(user)
                inputs = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
                prompt_length = inputs["input_ids"].shape[1]
                if prompt_length + self.args.max_new_tokens <= self.context_length:
                    break
                del inputs
                if not session.turns:
                    raise HTTPException(413, "This turn exceeds the model context. Try a shorter recording.")
                session.turns.pop(0).directory.cleanup()
            inputs = inputs.to(self.device)
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=self.args.max_new_tokens,
                    return_dict_in_generate=True, do_sample=False,
                    generation_mode="interleaved", audio_temperature=0.4,
                    audio_repetition_penalty=1.2,
                )
                end = int(outputs.sequence_lengths[0])
                text = self.processor.decode(
                    outputs.sequences[0, prompt_length:end], skip_special_tokens=True,
                ).replace("<think></think>", "").strip()
                audio_length = int(outputs.audio_lengths[0])
                if audio_length < 1:
                    raise RuntimeError("The model returned no audio. Try increasing --max-new-tokens.")
                waveform = self.codec.decode(
                    outputs.audio_codes[:1, :, :audio_length]
                ).audio_values[0]
            import numpy as np

            samples = waveform.detach().float().cpu().numpy().reshape(-1)
            if samples.size == 0 or not np.isfinite(samples).all():
                raise RuntimeError("The audio decoder returned an invalid waveform.")
            pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(OUTPUT_RATE)
                wav.writeframes(pcm)
            session.turns.append(Turn(directory, user, {"role": "assistant", "content": text}))
            directory = None  # The session owns the files until this turn expires.
            while len(session.turns) > self.args.history_turns:
                session.turns.pop(0).directory.cleanup()
            return {
                "text": text, "audio": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "sample_rate": OUTPUT_RATE, "seconds": round(time.monotonic() - started, 2),
                "frames": len(frames),
            }
        except HTTPException:
            raise
        except Exception as exc:
            LOG.exception("Generation failed")
            if hasattr(self, "torch") and self.device.startswith("cuda"):
                self.torch.cuda.empty_cache()
            raise HTTPException(500, f"Model inference failed: {exc}") from exc
        finally:
            if directory is not None:
                directory.cleanup()
            self.lock.release()


def create_app(args, engine=None):
    engine = engine or Engine(args)

    @asynccontextmanager
    async def lifespan(app):
        if not args.ui_only:
            try:
                await run_in_threadpool(engine.load)
            except Exception:
                LOG.exception("Could not load model. Finish the download, check dependencies, or use --model PATH.")
                raise
        yield
        engine.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.engine = engine

    @app.get("/", response_class=FileResponse)
    def index():
        return FileResponse(Path(__file__).with_name("index.html"), media_type="text/html",
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status():
        return {"ready": engine.ready, "model": args.model, "device": engine.device,
                "max_seconds": MAX_SECONDS, "max_frames": MAX_FRAMES}

    @app.post("/api/turn")
    async def turn(request: Request):
        # Read with a cap even when Content-Length is absent.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY:
                raise HTTPException(413, "Recording is too large.")
        import json

        try:
            payload = json.loads(body)
            validated = await run_in_threadpool(validate_turn, payload)
        except (ValueError, TypeError, UnicodeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return JSONResponse(await run_in_threadpool(engine.respond, *validated))

    @app.delete("/api/session/{session_id}")
    async def reset(session_id: str):
        await run_in_threadpool(engine.reset, session_id)
        return {"ok": True}

    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="LiquidAI/LFM3-Omni-2B", help="Hugging Face ID or local model directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--history-turns", type=int, default=3, help="Past camera/audio turns retained per conversation")
    parser.add_argument("--download", action="store_true", help="Allow downloading missing model files")
    parser.add_argument("--ui-only", action="store_true", help="Serve the interface without loading the model")

    args = parser.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 8192:
        parser.error("--max-new-tokens must be between 1 and 8192")
    if not 0 <= args.history_turns <= 8:
        parser.error("--history-turns must be between 0 and 8")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    LOG.info("Open http://localhost:%s", args.port)
    uvicorn.run(create_app(args), host=args.host, port=args.port)
