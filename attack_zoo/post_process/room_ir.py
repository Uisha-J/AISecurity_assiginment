"""Room impulse response convolution — simulates recording in a real room."""

from __future__ import annotations
import random
from pathlib import Path
from typing import Optional

import numpy as np

from ..base import PostProcessor


class RoomIRPostProcessor(PostProcessor):
    name = "room_ir"

    def __init__(self, rir_dir: str, rng: Optional[random.Random] = None) -> None:
        self.rir_dir = Path(rir_dir)
        self.rng = rng or random.Random()
        self._rir_files: list[Path] = []

    def is_available(self) -> bool:
        if not self.rir_dir.exists():
            return False
        if not self._rir_files:
            self._rir_files = sorted(self.rir_dir.rglob("*.wav"))
        return bool(self._rir_files)

    def apply(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.is_available():
            return waveform
        import soundfile as sf
        from scipy.signal import fftconvolve

        rir_path = self.rng.choice(self._rir_files)
        rir, rir_sr = sf.read(str(rir_path), dtype="float32")
        if rir.ndim > 1:
            rir = rir.mean(axis=1)
        if rir_sr != sample_rate:
            # cheap nearest-rate handling; for production use librosa.resample
            import librosa
            rir = librosa.resample(rir, orig_sr=rir_sr, target_sr=sample_rate)
        rir = rir / (np.abs(rir).max() + 1e-8)

        out = fftconvolve(waveform, rir, mode="full")[: len(waveform)]
        # Loudness preserve
        peak = np.abs(out).max() + 1e-8
        out = out / peak * np.abs(waveform).max()
        return out.astype(np.float32)
