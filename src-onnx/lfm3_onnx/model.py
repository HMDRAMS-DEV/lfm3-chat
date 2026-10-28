"""Omni encoders, selective bundle downloads, and streamed reply audio."""

import json
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from .generation import AudioSampling, ONNXAudioModel
from .preprocessing import image_patches

MODEL_ID = "LiquidAI/LFM3-Omni-2B-ONNX"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "pyproject.toml").is_file():
    PROJECT_ROOT = Path.cwd()
MODEL_DIR = PROJECT_ROOT / "resources/models/lfm3-omni-2b-onnx"

BUNDLES = {
    "q4": {
        "decoder": "decoder_q4.onnx", "vision_encoder": "vision_encoder_q8.onnx",
        "audio_encoder": "audio_encoder.onnx", "depth_decoder": "depth_decoder.onnx",
        "codec_decoder": "codec_decoder.onnx",
    },
    "q4f16": {
        "decoder": "decoder_q4f16.onnx", "vision_encoder": "vision_encoder_fp16.onnx",
        "audio_encoder": "audio_encoder_fp16.onnx", "depth_decoder": "depth_decoder.onnx",
        "codec_decoder": "codec_decoder_fp16.onnx",
    },
}


def choose_device_precision(device, precision, available):
    if device == "auto":
        device = "cpu" if precision == "q4" else (
            "cuda" if "CUDAExecutionProvider" in available else "cpu"
        )
    if precision == "auto":
        precision = "q4f16" if device == "cuda" else "q4"
    expected = "q4f16" if device == "cuda" else "q4"
    if precision != expected:
        raise ValueError(f"Use --precision {expected} with --device {device}.")
    if device == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError("CUDA is unavailable. Install the gpu extra on the GPU machine.")
    return device, precision


def bundle_patterns(precision):
    return ["*.json", "*.jinja", "LICENSE*", "README.md",
            *(f"onnx/{name}*" for name in BUNDLES[precision].values())]


def resolve_model(model, precision, download=False):
    from huggingface_hub import snapshot_download

    path = Path(model).expanduser()
    if path.is_dir():
        # Explicit local folders never cause a network request.
        return path
    if model != MODEL_ID:
        raise FileNotFoundError(f"Model folder not found: {model}")
    if download:
        snapshot_download(MODEL_ID, local_dir=MODEL_DIR, allow_patterns=bundle_patterns(precision))
        return MODEL_DIR
    required = [MODEL_DIR / "onnx" / name for name in BUNDLES[precision].values()]
    if all(path.is_file() for path in required):
        return MODEL_DIR
    try:
        cached = Path(snapshot_download(MODEL_ID, local_files_only=True))
        if all((cached / "onnx" / name).is_file() for name in BUNDLES[precision].values()):
            return cached
    except OSError:
        pass
    raise FileNotFoundError(
        f"The {precision} bundle is not cached. Start once with --download, "
        f"or supply --model /path/to/LFM3-Omni-2B-ONNX. Downloads stay in {MODEL_DIR}."
    )


class OmniModel(ONNXAudioModel):
    bundles = BUNDLES

    def load_preprocessing(self):
        super().load_preprocessing()
        self.image_config = json.loads((self.model_dir / "processor_config.json").read_text())["image_processor"]
        self.image_token = json.loads((self.model_dir / "config.json").read_text())["image_token_id"]

    def prepare(self, messages, check=lambda: None):
        audio_features, image_features = [], []
        audio_counts, image_counts = [], []
        for message in messages:
            content = message["content"]
            if not isinstance(content, list):
                continue
            for item in content:
                check()
                if item["type"] == "audio":
                    with wave.open(item["audio"], "rb") as wav:
                        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                            raise ValueError("Expected PCM16 mono 16 kHz audio.")
                        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
                    features = self.encode_audio([samples])
                    audio_features.append(features)
                    audio_counts.append(len(features))
                elif item["type"] == "image":
                    with Image.open(item["image"]) as image:
                        blocks, count = image_patches(image, self.image_config)
                    features = self.vision_encoder.run(None, {"pixel_values": blocks})[0]
                    image_features.append(features)
                    image_counts.append(count)
        ids = self.prompt_ids(messages, audio_counts)
        if ids.count(self.image_token) != len(image_counts):
            raise ValueError("Camera placeholder count does not match the images.")
        counts = iter(image_counts)
        ids = [value for token in ids for value in (
            [token] * next(counts) if token == self.image_token else [token]
        )]
        check()
        return {
            "input_ids": ids,
            "audio_features": np.concatenate(audio_features) if audio_features else None,
            "image_features": np.concatenate(image_features) if image_features else None,
        }

    def reply(self, inputs, max_new_tokens, emit, check):
        text_ids, last_text = [], ""
        stream = CodecStream(self, emit, check) if emit else None

        def on_token(token):
            nonlocal last_text
            check()
            text_ids.append(token)
            text = self.tokenizer.decode(text_ids).replace("<think></think>", "").strip()
            if emit and text != last_text and not text.endswith("\ufffd"):
                last_text = text
                emit({"type": "text", "text": text})

        def on_frame(frame):
            check()
            if stream:
                stream.put(frame)

        check()
        result = self.generate_ids(
            **inputs, max_new_tokens=max_new_tokens, generation_mode="interleaved",
            audio_sampling=AudioSampling(temperature=0.4, repetition_penalty=1.2),
            on_token=on_token, on_frame=on_frame,
        )
        check()
        if not len(result.audio_codes):
            raise RuntimeError("The model returned no speech. Try increasing --max-new-tokens.")
        pcm = stream.end() if stream else waveform_pcm(self.decode_audio(result.audio_codes))
        text = self.tokenizer.decode(result.tokens).replace("<think></think>", "").strip()
        return text, pcm


def waveform_pcm(samples):
    samples = np.asarray(samples).reshape(-1)
    if not np.isfinite(samples).all():
        raise RuntimeError("The audio decoder returned an invalid waveform.")
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


class CodecStream:
    """Re-decode accumulated codes, retaining the inverse-STFT overlap until stable.

    The published codec graph has no cache inputs. Recomputing the prefix costs
    more than the PyTorch streaming codec, but allows playback before turn end.
    """

    def __init__(self, model, emit, check):
        self.model, self.emit, self.check = model, emit, check
        self.codes, self.parts = [], []
        self.offset = 0
        self.wave = None

    def put(self, codes):
        self.check()
        self.codes.append(np.array(codes, copy=True))
        if len(self.codes) % 4 == 0:
            self.flush(final=False)

    def flush(self, final):
        self.check()
        if self.wave is None or len(self.codes) % 4 or not final:
            self.wave = self.model.decode_audio(np.stack(self.codes))
        self.check()
        # Causal codec features are stable; only the inverse STFT right edge
        # changes when additional frames arrive. Retain a full FFT of samples.
        limit = len(self.wave) if final else max(0, len(self.wave) - self.model.config.codec_n_fft)
        pcm = waveform_pcm(self.wave[self.offset:limit])
        self.offset = limit
        if pcm:
            self.parts.append(pcm)
            self.emit({"type": "audio", "pcm": pcm})

    def end(self):
        self.flush(final=True)
        return b"".join(self.parts)
