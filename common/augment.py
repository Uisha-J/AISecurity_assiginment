"""Training-time augmentation chain.

Three big knobs that determine real-world EER:
1. Codec round-trip   — single biggest factor in telephony deployment
2. Room IR convolution + additive noise (MUSAN-style)
3. RawBoost (Tak et al., Interspeech 2022) — anti-spoofing-specific augmentation

Each augmentation is applied with its own probability so the network sees a
broad distribution. We deliberately keep magnitudes wide.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import signal as scisig


# Reuse post-processors as augmentations (post_process is sample-time;
# augment is train-time but semantically identical).
from ..attack.zoo.post_process.codec import CodecPostProcessor
from ..attack.zoo.post_process.room_ir import RoomIRPostProcessor
from ..attack.zoo.post_process.noise_mix import NoisePostProcessor


# ============================================================== RawBoost
class RawBoost:
    """RawBoost augmentation (Tak et al., 2022).

    Three families; we randomly pick one when called:
      1. Linear convolutive noise (band-limited convolution)
      2. Impulsive signal-dependent additive noise
      3. Stationary signal-independent additive noise

    Designed specifically to harden spoof detectors against unknown
    channel/device variation.
    """

    def __init__(
        self,
        algos: Sequence[int] = (1, 2, 3),
        rng: Optional[random.Random] = None,
    ) -> None:
        self.algos = list(algos)
        self.rng = rng or random.Random()

    def __call__(self, x: np.ndarray, sr: int) -> np.ndarray:
        algo = self.rng.choice(self.algos)
        if algo == 1:
            return self._linear_conv(x, sr)
        elif algo == 2:
            return self._impulsive(x, sr)
        else:
            return self._stationary(x, sr)

    def _linear_conv(self, x: np.ndarray, sr: int) -> np.ndarray:
        n_taps = self.rng.choice([5, 9, 13])
        cutoff = self.rng.uniform(0.05, 0.45)
        b = scisig.firwin(n_taps, cutoff)
        y = scisig.lfilter(b, [1.0], x).astype(np.float32)
        # mix back to keep gain reasonable
        m = self.rng.uniform(0.3, 0.8)
        return ((1 - m) * x + m * y).astype(np.float32)

    def _impulsive(self, x: np.ndarray, sr: int) -> np.ndarray:
        n = len(x)
        n_imp = int(self.rng.uniform(0.001, 0.01) * n)
        idx = np.random.randint(0, n, size=n_imp)
        amp = (np.random.rand(n_imp) * 2 - 1) * np.abs(x).max() * self.rng.uniform(0.3, 1.0)
        y = x.copy()
        y[idx] = y[idx] + amp
        peak = np.abs(y).max()
        return (y / max(peak, 1.0)).astype(np.float32)

    def _stationary(self, x: np.ndarray, sr: int) -> np.ndarray:
        snr_db = self.rng.uniform(10, 30)
        noise = np.random.randn(len(x)).astype(np.float32)
        sig_p = (x ** 2).mean() + 1e-12
        n_p_target = sig_p / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(n_p_target / (noise.var() + 1e-12))
        y = x + noise
        peak = np.abs(y).max()
        return (y / max(peak, 1.0)).astype(np.float32)


# ============================================================== combined chain
@dataclass
class AugmentChainSpec:
    """Subset of configs/default.yaml::augmentation."""
    codec_enabled: bool = True
    codec_prob: float = 0.5
    codec_codecs: Sequence[str] = ("opus", "mp3", "g711_alaw", "g711_ulaw")

    rir_enabled: bool = True
    rir_prob: float = 0.3
    rir_dir: str = "data/augment/rirs"

    noise_enabled: bool = True
    noise_prob: float = 0.4
    noise_dir: str = "data/augment/musan"
    noise_snr_db: tuple[float, float] = (5.0, 20.0)

    rawboost_enabled: bool = True
    rawboost_prob: float = 0.5
    rawboost_algos: Sequence[int] = (1, 2, 3)


def build_augmentation_chain(
    spec: AugmentChainSpec,
    rng: Optional[random.Random] = None,
) -> Callable[[np.ndarray, int], np.ndarray]:
    rng = rng or random.Random()
    codec = CodecPostProcessor(codecs=spec.codec_codecs, rng=rng) if spec.codec_enabled else None
    rir = RoomIRPostProcessor(spec.rir_dir, rng=rng) if spec.rir_enabled else None
    noise = NoisePostProcessor(spec.noise_dir, spec.noise_snr_db, rng=rng) if spec.noise_enabled else None
    rb = RawBoost(spec.rawboost_algos, rng=rng) if spec.rawboost_enabled else None

    def _apply(x: np.ndarray, sr: int) -> np.ndarray:
        # RIR first (acoustic), then noise (acoustic), then codec (channel),
        # then RawBoost (electronic distortion).
        if rir is not None and rng.random() < spec.rir_prob:
            x = rir.apply(x, sr)
        if noise is not None and rng.random() < spec.noise_prob:
            x = noise.apply(x, sr)
        if codec is not None and rng.random() < spec.codec_prob:
            x = codec.apply(x, sr)
        if rb is not None and rng.random() < spec.rawboost_prob:
            x = rb(x, sr)
        return x

    return _apply
