"""Always-available artifact voice-conversion attack generator.

This generator takes an authorized source wav and applies signal-level
transformations that resemble common VC/replay artifacts: pitch/formant drift,
band limiting, phase distortion, resampling, and quantization. It gives the
defense a concrete attack family even when RVC/seed-vc weights are absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy import signal

from ..base import AttackGenerator, AttackMetadata, AttackResult


class ArtifactVCAttack(AttackGenerator):
    """Signal-level VC/replay spoof generator."""

    name = "artifact_vc"
    family = "vc"

    def __init__(
        self,
        sample_rate: int = 16000,
        segment_seconds: float = 4.0,
        pitch_shift_steps: tuple[int, ...] = (-2, -1, 1, 2),
    ) -> None:
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.pitch_shift_steps = pitch_shift_steps

    def is_available(self) -> bool:
        return True

    def _load_source(self, path: str) -> np.ndarray:
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        n = int(self.sample_rate * self.segment_seconds)
        if len(wav) >= n:
            start = (len(wav) - n) // 2
            wav = wav[start : start + n]
        else:
            reps = int(np.ceil(n / max(len(wav), 1)))
            wav = np.tile(wav, reps)[:n]
        return wav.astype(np.float32)

    @staticmethod
    def _safe(x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        if peak > 1.0:
            x = x / peak
        return np.clip(x, -1.0, 1.0).astype(np.float32)

    def _convert(self, wav: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        import librosa

        steps = int(rng.choice(self.pitch_shift_steps))
        y = librosa.effects.pitch_shift(wav, sr=self.sample_rate, n_steps=steps)

        # Resample down and back to create bandwidth and phase artifacts.
        mid_sr = int(rng.choice([8000, 11025, 12000]))
        y = librosa.resample(y, orig_sr=self.sample_rate, target_sr=mid_sr)
        y = librosa.resample(y, orig_sr=mid_sr, target_sr=self.sample_rate)
        if len(y) < len(wav):
            y = np.pad(y, (0, len(wav) - len(y)))
        y = y[: len(wav)]

        # Spectral tilt/formant-ish filtering.
        cutoff = float(rng.uniform(0.18, 0.42))
        taps = signal.firwin(int(rng.choice([17, 31, 47])), cutoff)
        filtered = signal.lfilter(taps, [1.0], y).astype(np.float32)
        mix = float(rng.uniform(0.35, 0.8))
        y = (1.0 - mix) * y + mix * filtered

        # Coarse quantization and low noise floor, similar to lossy synthesis.
        levels = int(rng.choice([128, 256, 512]))
        y = np.round((y + 1.0) * (levels / 2.0)) / (levels / 2.0) - 1.0
        noise = rng.standard_normal(len(y)).astype(np.float32)
        y = y + noise * float(rng.uniform(0.001, 0.006))

        meta = {
            "pitch_shift_steps": steps,
            "roundtrip_sample_rate": mid_sr,
            "filter_cutoff_norm": cutoff,
            "filter_mix": mix,
            "quantization_levels": levels,
        }
        return self._safe(y), meta

    def generate(
        self,
        *,
        text: Optional[str] = None,
        reference_wav: Optional[str] = None,
        target_wav: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> AttackResult:
        source = target_wav or reference_wav
        if source is None:
            raise ValueError("ArtifactVCAttack requires target_wav or reference_wav")
        if not Path(source).exists():
            raise FileNotFoundError(source)

        rng = np.random.default_rng(seed)
        wav = self._load_source(source)
        converted, extra = self._convert(wav, rng)

        return AttackResult(
            waveform=converted,
            sample_rate=self.sample_rate,
            metadata=AttackMetadata(
                algorithm=self.name,
                family=self.family,
                reference_wav=reference_wav,
                text=text,
                extra={
                    "content_wav": target_wav or source,
                    "note": "safe baseline artifact VC; no model-based speaker cloning",
                    **extra,
                },
            ),
        )
