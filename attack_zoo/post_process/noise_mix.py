"""Additive noise mixing — MUSAN-style backgrounds at controlled SNR."""

from __future__ import annotations
import random
from pathlib import Path
from typing import Optional

import numpy as np

from ..base import PostProcessor


class NoisePostProcessor(PostProcessor):
    name = "noise"

    def __init__(
        self,
        noise_dir: str,
        snr_db_range: tuple[float, float] = (5.0, 20.0),
        rng: Optional[random.Random] = None,
    ) -> None:
        self.noise_dir = Path(noise_dir)
        self.snr_range = snr_db_range
        self.rng = rng or random.Random()
        self._files: list[Path] = []

    def is_available(self) -> bool:
        if not self.noise_dir.exists():
            return False
        if not self._files:
            self._files = sorted(self.noise_dir.rglob("*.wav"))
        return bool(self._files)

    def apply(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.is_available():
            return waveform
        import soundfile as sf
        path = self.rng.choice(self._files)
        noise, sr = sf.read(str(path), dtype="float32")
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        if sr != sample_rate:
            import librosa
            noise = librosa.resample(noise, orig_sr=sr, target_sr=sample_rate)

        # Loop or crop noise to match length
        if len(noise) < len(waveform):
            reps = int(np.ceil(len(waveform) / max(len(noise), 1)))
            noise = np.tile(noise, reps)
        start = self.rng.randint(0, max(len(noise) - len(waveform), 0))
        noise = noise[start : start + len(waveform)]

        # SNR scaling
        snr_db = self.rng.uniform(*self.snr_range)
        sig_p = np.mean(waveform ** 2) + 1e-12
        noise_p = np.mean(noise ** 2) + 1e-12
        target_noise_p = sig_p / (10 ** (snr_db / 10.0))
        noise = noise * np.sqrt(target_noise_p / noise_p)

        out = waveform + noise
        peak = np.abs(out).max()
        if peak > 1.0:
            out = out / peak
        return out.astype(np.float32)
