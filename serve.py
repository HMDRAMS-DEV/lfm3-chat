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

For another computer, use HTTPS with --host 0.0.0.0 --ssl-certfile CERT
--ssl-keyfile KEY, or tunnel localhost. Browsers require a secure context for
camera/microphone access. This server is intended for your local machine.

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
from fastapi.responses import HTMLResponse, JSONResponse
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

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(HTML, headers={"Cache-Control": "no-store"})

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


HTML = r'''<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LFM3 video chat</title>
<style>
:root { color-scheme: dark; font: 16px/1.5 system-ui, sans-serif; background: #101318; color: #edf1f7 }
* { box-sizing: border-box } body { margin: 0; padding: 28px; max-width: 1200px; margin-inline: auto }
h1 { font-size: 26px; margin: 0 } .muted { color: #a4aebd } header { margin-bottom: 22px }
main { display: grid; grid-template-columns: 1fr 1fr; gap: 22px } .panel { background: #191f28; border: 1px solid #303947; border-radius: 16px; padding: 20px }
video { width: 100%; aspect-ratio: 4/3; background: #090c10; border-radius: 10px; object-fit: contain; transform: scaleX(-1) }
.buttons { display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0 } button { font: inherit; font-weight: 600; padding: 10px 16px; border: 0; border-radius: 8px; cursor: pointer; background: #a8e6c5; color: #10271d }
button.secondary { background: #313e50; color: #edf1f7 } button:disabled { opacity: .4; cursor: default } button.active { background: #ff9b98; color: #301414 }
label { display: block; margin-block: 12px } input[type=range] { width: 130px; vertical-align: middle }
#status { min-height: 28px; font-weight: 600 } #status.error { color: #ffaaa7 } meter { width: 100%; height: 12px }
#messages { height: 400px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; padding-bottom: 8px }
.message { border-radius: 10px; padding: 12px 14px; background: #263449; white-space: pre-wrap; overflow-wrap: anywhere }
.message.user { background: #263c34; align-self: end } .meta { color: #b3bfd0; font-size: 13px; margin-bottom: 5px }
audio { max-width: 100%; margin-top: 14px } #play { display: none } .hint { font-size: 14px; color: #a4aebd }
@media(max-width: 760px) { body { padding: 16px } main { grid-template-columns: 1fr } #messages { height: 260px } }
</style>
<header><h1>LFM3 video chat</h1><div class="muted">LiquidAI / LFM3-Omni-2B</div></header>
<main>
  <section class="panel" aria-label="Camera and microphone">
    <video id="camera" autoplay playsinline muted></video>
    <div class="buttons">
      <button id="start">Start camera &amp; mic</button>
      <button id="stop" class="secondary" disabled>Stop</button>
      <button id="talk" class="secondary" disabled>Speak</button>
    </div>
    <div id="status" role="status" aria-live="polite">Checking model...</div>
    <meter id="level" min="0" max="0.12" value="0" aria-label="Microphone level"></meter>
    <label><input id="automatic" type="checkbox" checked> Send automatically after a pause</label>
    <label><input id="useCamera" type="checkbox" checked> Include camera frames</label>
    <label class="hint">Mic threshold <input id="threshold" type="range" min="0.005" max="0.08" step="0.001" value="0.018"></label>
    <p class="hint">Speak naturally, then pause for about a second. You can also use Speak / Send. Each turn includes up to six camera frames. The mic pauses while the model answers.</p>
    <p class="hint">Video preview is mirrored. The model sees the original camera image.</p>
  </section>
  <section class="panel" aria-label="Conversation">
    <div id="messages" role="log" aria-live="polite"><p class="muted" id="empty">Your spoken replies will appear here.</p></div>
    <audio id="reply" controls></audio>
    <button id="play">Play reply</button>
    <div class="buttons"><button id="clear" class="secondary">New conversation</button></div>
    <p class="hint" id="modelInfo"></p>
  </section>
</main>
<script>
'use strict';
const $ = id => document.getElementById(id);
let sessionId = crypto.randomUUID();
let stream, context, source, worklet, mute, timer, audioUrl;
let connected = false, busy = false, recording = false, manual = false;
let chunks = [], preRoll = [], frames = [], recordedSamples = 0;
let captureRate = 48000, lastVoice = 0, epoch = 0, resumeAt = 0, modelReady = false;
let maxSeconds = 30, maxFrames = 6;

function status(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}
function message(text, who, meta = '') {
  $('empty')?.remove();
  const box = document.createElement('div'); box.className = 'message ' + who;
  if (meta) { const label = document.createElement('div'); label.className = 'meta'; label.textContent = meta; box.append(label); }
  const body = document.createElement('div'); body.textContent = text; box.append(body);
  $('messages').append(box); $('messages').scrollTop = $('messages').scrollHeight;
}
function controls() {
  $('start').disabled = connected || busy || !modelReady;
  $('stop').disabled = !connected;
  $('talk').disabled = !connected || busy || !$('reply').paused;
  $('clear').disabled = busy;
  $('talk').textContent = recording ? 'Send' : 'Speak';
  $('talk').classList.toggle('active', recording);
}
function listening() {
  if (connected && !busy && $('reply').paused) {
    status($('automatic').checked ? 'Listening...' : 'Ready. Click Speak to record.');
  }
  controls();
}
function clearRecording() {
  recording = false; manual = false; chunks = []; frames = []; recordedSamples = 0; preRoll = [];
  controls();
}
function captureFrame() {
  const video = $('camera');
  if (!recording || !$('useCamera').checked || !video.videoWidth) return;
  const canvas = document.createElement('canvas');
  const scale = Math.min(1, 384 / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  frames.push({time: recordedSamples / captureRate, image: canvas.toDataURL('image/jpeg', 0.75).split(',')[1]});
}
function beginTurn(isManual) {
  if (!connected || busy || recording || !$('reply').paused || performance.now() < resumeAt) return;
  recording = true; manual = isManual; chunks = preRoll.slice(); preRoll = [];
  recordedSamples = chunks.reduce((n, c) => n + c.length, 0); frames = [];
  lastVoice = performance.now(); captureFrame(); status('Recording...'); controls();
}
function onSamples(samples) {
  let sum = 0; for (const x of samples) sum += x * x;
  const rms = Math.sqrt(sum / samples.length); $('level').value = rms;
  if (!connected || busy || !$('reply').paused || performance.now() < resumeAt) { preRoll = []; return; }
  const now = performance.now(), voiced = rms >= Number($('threshold').value);
  if (!recording) {
    if ($('automatic').checked && voiced) beginTurn(false);
    else {
      preRoll.push(samples);
      while (preRoll.length > Math.ceil(captureRate * 0.25 / samples.length)) preRoll.shift();
      return;
    }
  }
  chunks.push(samples); recordedSamples += samples.length;
  if (voiced) lastVoice = now;
  if (recordedSamples >= captureRate * (maxSeconds - 0.1) ||
      (!manual && now - lastVoice >= 1000 && recordedSamples > captureRate * 0.5)) {
    void sendTurn();
  }
}
function selectFrames(all) {
  if (all.length <= maxFrames) return all;
  return Array.from({length: maxFrames}, (_, i) => all[Math.round(i * (all.length - 1) / (maxFrames - 1))]);
}
function bytesBase64(bytes) {
  let text = '';
  for (let i = 0; i < bytes.length; i += 16384) text += String.fromCharCode(...bytes.subarray(i, i + 16384));
  return btoa(text);
}
async function encodeWav(parts, rate) {
  const size = parts.reduce((n, c) => n + c.length, 0);
  const offline = new OfflineAudioContext(1, Math.ceil(size * 16000 / rate), 16000);
  const buffer = offline.createBuffer(1, size, rate), data = buffer.getChannelData(0);
  let at = 0; for (const part of parts) { data.set(part, at); at += part.length; }
  const node = offline.createBufferSource(); node.buffer = buffer; node.connect(offline.destination); node.start();
  const pcm = (await offline.startRendering()).getChannelData(0);
  const wav = new ArrayBuffer(44 + pcm.length * 2), view = new DataView(wav);
  const word = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
  word(0, 'RIFF'); view.setUint32(4, 36 + pcm.length * 2, true); word(8, 'WAVE'); word(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  word(36, 'data'); view.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 32767, true);
  return bytesBase64(new Uint8Array(wav));
}
async function sendTurn() {
  if (!recording || busy) return;
  if (recordedSamples < captureRate * 0.1) { clearRecording(); listening(); return; }
  captureFrame();
  const parts = chunks, selected = $('useCamera').checked ? selectFrames(frames) : [];
  const duration = recordedSamples / captureRate, thisEpoch = epoch;
  busy = true; clearRecording(); status('Model is thinking...'); controls();
  message(`Spoken message · ${duration.toFixed(1)}s · ${selected.length} camera frames`, 'user');
  try {
    const audio = await encodeWav(parts, captureRate);
    if (epoch !== thisEpoch) return;
    const response = await fetch('/api/turn', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, audio, frames: selected})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'The request failed.');
    if (epoch !== thisEpoch) return;
    message(result.text || '[Audio reply]', 'assistant', `LFM3 · ${result.seconds}s`);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    const raw = atob(result.audio), bytes = Uint8Array.from(raw, c => c.charCodeAt(0));
    audioUrl = URL.createObjectURL(new Blob([bytes], {type: 'audio/wav'}));
    $('reply').src = audioUrl;
    try { await $('reply').play(); status('Speaking...'); }
    catch { status('Click Play reply to hear the answer.'); $('play').style.display = 'inline-block'; }
  } catch (error) {
    if (epoch === thisEpoch) { status(error.message, true); message(error.message, 'error'); }
  } finally {
    busy = false; resumeAt = performance.now() + 500; controls();
    // Preserve errors and autoplay instructions until the next interaction.
  }
}

const workletCode = `
class MicCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buffer = new Float32Array(2048); this.at = 0; }
  process(inputs) {
    const channels = inputs[0];
    if (channels && channels.length) {
      for (let i = 0; i < channels[0].length; i++) {
        let value = 0; for (const channel of channels) value += channel[i];
        this.buffer[this.at++] = value / channels.length;
        if (this.at === this.buffer.length) {
          this.port.postMessage(this.buffer, [this.buffer.buffer]);
          this.buffer = new Float32Array(2048); this.at = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('mic-capture', MicCapture);`;

$('start').onclick = async () => {
  if (!window.isSecureContext || !navigator.mediaDevices) {
    status('Camera and mic require localhost or HTTPS.', true); return;
  }
  $('start').disabled = true; status('Requesting camera and microphone...');
  try {
    // Resume audio inside the click handler before waiting for permissions.
    context = new AudioContext(); await context.resume();
    stream = await navigator.mediaDevices.getUserMedia({
      video: $('useCamera').checked ? {width: {ideal: 640}, height: {ideal: 480}} : false,
      audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}
    });
    $('camera').srcObject = stream;
    captureRate = context.sampleRate;
    const moduleUrl = URL.createObjectURL(new Blob([workletCode], {type: 'text/javascript'}));
    try { await context.audioWorklet.addModule(moduleUrl); } finally { URL.revokeObjectURL(moduleUrl); }
    source = context.createMediaStreamSource(stream);
    worklet = new AudioWorkletNode(context, 'mic-capture');
    worklet.port.onmessage = event => onSamples(event.data);
    mute = context.createGain(); mute.gain.value = 0;
    source.connect(worklet); worklet.connect(mute); mute.connect(context.destination);
    connected = true; epoch++; timer = setInterval(captureFrame, 500);
    for (const track of stream.getTracks()) track.onended = () => { void stopMedia(); status('Camera or microphone disconnected.', true); };
    listening();
  } catch (error) {
    await stopMedia(); status('Could not start camera/mic: ' + error.message, true);
  }
};
async function stopMedia() {
  connected = false; epoch++; clearInterval(timer);
  if (worklet) { worklet.port.onmessage = null; worklet.disconnect(); }
  source?.disconnect(); mute?.disconnect();
  for (const track of stream?.getTracks() || []) { track.onended = null; track.stop(); }
  stream = null; $('camera').srcObject = null;
  if (context && context.state !== 'closed') await context.close();
  context = null; source = null; worklet = null; mute = null;
  $('reply').pause(); $('play').style.display = 'none';
  clearRecording(); $('level').value = 0; controls();
}
$('stop').onclick = async () => { await stopMedia(); status('Stopped.'); };
$('talk').onclick = () => { if (recording) void sendTurn(); else beginTurn(true); };
$('automatic').onchange = () => { if (!recording) listening(); };
$('useCamera').onchange = () => {
  if (!$('useCamera').checked) frames = [];
  if (connected && $('useCamera').checked && !stream.getVideoTracks().length) status('Stop and restart to enable the camera.');
};
$('reply').onplay = () => { preRoll = []; clearRecording(); status('Speaking...'); controls(); };
$('reply').onpause = $('reply').onended = () => {
  $('play').style.display = 'none'; preRoll = []; resumeAt = performance.now() + 500; listening();
};
$('play').onclick = async () => {
  try { await $('reply').play(); $('play').style.display = 'none'; }
  catch (error) { status('Audio playback failed: ' + error.message, true); }
};
$('clear').onclick = async () => {
  if (busy) return;
  busy = true; clearRecording(); $('reply').pause(); controls();
  try {
    const response = await fetch('/api/session/' + sessionId, {method: 'DELETE'});
    if (!response.ok) throw new Error((await response.json()).detail);
    sessionId = crypto.randomUUID(); epoch++;
    $('messages').replaceChildren(); $('reply').removeAttribute('src'); $('reply').load();
    if (audioUrl) URL.revokeObjectURL(audioUrl); audioUrl = null;
    status(connected ? ($('automatic').checked ? 'Listening...' : 'Ready. Click Speak to record.') : 'Conversation cleared.');
  } catch (error) { status(error.message, true); }
  finally { busy = false; resumeAt = performance.now() + 500; controls(); }
};
window.addEventListener('pagehide', () => {
  for (const track of stream?.getTracks() || []) track.stop();
  if (audioUrl) URL.revokeObjectURL(audioUrl);
});
fetch('/api/status').then(r => r.json()).then(info => {
  modelReady = info.ready; maxSeconds = info.max_seconds; maxFrames = info.max_frames;
  $('modelInfo').textContent = info.model + ' · ' + info.device;
  status(info.ready ? 'Ready. Start your camera and microphone.' : 'UI-only mode. Restart without --ui-only to chat.');
  controls();
}).catch(error => status('Cannot reach server: ' + error.message, true));
</script>
</html>'''


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
    parser.add_argument("--ssl-certfile")
    parser.add_argument("--ssl-keyfile")
    args = parser.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 8192:
        parser.error("--max-new-tokens must be between 1 and 8192")
    if not 0 <= args.history_turns <= 8:
        parser.error("--history-turns must be between 0 and 8")
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("Supply both --ssl-certfile and --ssl-keyfile")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    LOG.info("Open %s://localhost:%s", "https" if args.ssl_certfile else "http", args.port)
    uvicorn.run(create_app(args), host=args.host, port=args.port,
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)
