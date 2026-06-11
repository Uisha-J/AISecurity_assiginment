"""Always-available synthetic TTS attack generator.

This generator is intentionally lightweight: it creates speech-like harmonic
audio from text with deterministic syllable envelopes, stable pitch, and
vocoder-style artifacts. It is useful as a safe baseline attack family for
testing the defense pipeline when heavyweight voice-cloning models are not
installed.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
from scipy import signal

from ..base import AttackGenerator, AttackMetadata, AttackResult


class SyntheticTTSAttack(AttackGenerator):
    """Text-conditioned synthetic spoof with TTS-like artifacts."""

    name = "synthetic_tts"
    family = "tts"

    def __init__(
        self,
        sample_rate: int = 16000,
        seconds_per_token: float = 0.22,
        min_seconds: float = 2.0,
        max_seconds: float = 6.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.seconds_per_token = seconds_per_token
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

    def is_available(self) -> bool:
        return True

    @staticmethod
    def _seed_from(text: str, seed: Optional[int]) -> int:
        if seed is not None:
            return seed
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little", signed=False)

    def generate(
        self,
        *,
        text: Optional[str] = None,
        reference_wav: Optional[str] = None,
        target_wav: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> AttackResult:
        text = text or "synthetic speech sample"
        rng = np.random.default_rng(self._seed_from(text, seed))

        tokens = max(1, len(text.split()))
        duration = np.clip(tokens * self.seconds_per_token, self.min_seconds, self.max_seconds)
        n = int(self.sample_rate * duration)
        t = np.arange(n, dtype=np.float32) / self.sample_rate

        f0 = rng.uniform(135.0, 205.0)
        # Too-stable pitch is a classic low-quality TTS tell.
        f0_curve = f0 * (1.0 + 0.006 * np.sin(2 * np.pi * rng.uniform(3.0, 5.0) * t))
        phase = 2 * np.pi * np.cumsum(f0_curve) / self.sample_rate

        wav = np.zeros(n, dtype=np.float32)
        for harmonic in range(1, 9):
            amp = 1.0 / (harmonic ** rng.uniform(0.8, 1.25))
            wav += amp * np.sin(harmonic * phase + rng.uniform(-0.04, 0.04))

        # Mechanical syllable envelope, keyed by the text content.
        syllable_hz = rng.uniform(3.0, 6.5)
        env = 0.55 + 0.45 * signal.square(2 * np.pi * syllable_hz * t, duty=0.58)
        env = signal.lfilter([0.04], [1.0, -0.96], env).astype(np.float32)
        env = env / max(float(env.max()), 1e-8)
        wav *= env

        # Fixed high-frequency comb/whistle artifact.
        whistle_hz = rng.choice([5200.0, 6100.0, 6900.0])
        wav += 0.035 * np.sin(2 * np.pi * whistle_hz * t)

        # Light quantization imitates a neural vocoder/postnet output floor.
        wav = wav / max(float(np.max(np.abs(wav))), 1e-8) * 0.65
        levels = 256
        wav = np.round((wav + 1.0) * (levels / 2.0)) / (levels / 2.0) - 1.0
        wav += 0.002 * rng.standard_normal(n).astype(np.float32)
        wav = wav / max(float(np.max(np.abs(wav))), 1e-8) * 0.6

        return AttackResult(
            waveform=wav.astype(np.float32),
            sample_rate=self.sample_rate,
            metadata=AttackMetadata(
                algorithm=self.name,
                family=self.family,
                reference_wav=reference_wav,
                text=text,
                extra={
                    "duration_seconds": float(duration),
                    "f0_hz": float(f0),
                    "note": "safe baseline synthetic TTS; no real speaker cloning",
                },
            ),
        )
