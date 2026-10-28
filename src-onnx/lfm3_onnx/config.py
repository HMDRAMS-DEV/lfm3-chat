"""Read inference settings from the checkpoint's JSON files."""

import json
from dataclasses import dataclass
from types import SimpleNamespace


@dataclass
class MelConfig:
    sampling_rate: int = 16000
    num_mel_bins: int = 128
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    log_floor: float = -12.0
    normalization_offset: float = 4.0
    normalization_divisor: float = 4.0

    def num_frames(self, samples):
        padded = max(self.n_fft, samples + (-(samples - self.n_fft) % self.hop_length))
        return 1 + (padded - self.n_fft) // self.hop_length

    @staticmethod
    def num_tokens(frames):
        return (frames + 7) // 8


def load_config(directory):
    raw = json.loads((directory / "config.json").read_text())
    processor = json.loads((directory / "processor_config.json").read_text())
    codec = json.loads((directory / "audio_tokenizer/config.json").read_text())
    feature = processor["feature_extractor"]
    tokens = SimpleNamespace(**{
        name: raw[f"{name}_token_id"]
        for name in ("audio", "audio_out", "audio_start", "mixed_start", "text_end")
    }, eos=raw["text_config"]["eos_token_id"])
    return SimpleNamespace(
        language=SimpleNamespace(**raw["text_config"]), tokens=tokens,
        mel=MelConfig(
            num_mel_bins=feature["feature_size"],
            **{key: feature[key] for key in MelConfig.__dataclass_fields__
               if key != "num_mel_bins" and key in feature},
        ),
        **{key: raw[key] for key in (
            "num_codebooks", "codebook_size", "audio_eos_code", "maskgit_schedule",
            "interleaved_n_text", "interleaved_n_audio",
        )},
        codec_n_fft=codec["stft_n_fft"], codec_hop_length=codec["stft_hop_length"],
    )
