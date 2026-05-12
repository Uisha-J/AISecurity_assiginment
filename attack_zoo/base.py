"""Base interfaces for attack zoo components.

All TTS, VC, and post-processing modules conform to one of two abstract bases:
- AttackGenerator: produce spoofed audio from text and/or reference voice
- PostProcessor:   transform an already-generated waveform in-place
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence
import json

import numpy as np


@dataclass
class AttackMetadata:
    """Provenance recorded with every spoofed sample."""
    algorithm: str               # e.g. "xtts_v2", "rvc"
    family: str                  # "tts" or "vc"
    reference_wav: Optional[str] = None    # path to speaker reference (VC source)
    text: Optional[str] = None             # source text (TTS)
    post_processing: list[str] = field(default_factory=list)  # codec/rir/noise applied
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class AttackResult:
    """Standardized output of any attack generator."""
    waveform: np.ndarray         # float32, mono, in range [-1, 1]
    sample_rate: int
    metadata: AttackMetadata

    def __post_init__(self) -> None:
        if self.waveform.ndim != 1:
            self.waveform = self.waveform.squeeze()
        if self.waveform.dtype != np.float32:
            self.waveform = self.waveform.astype(np.float32)


class AttackGenerator(ABC):
    """A TTS or VC system that produces spoofed audio."""

    #: short tag written into AttackMetadata.algorithm
    name: str = "abstract"
    #: "tts" or "vc"
    family: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool:
        """True if backing model + deps are installed."""

    @abstractmethod
    def generate(
        self,
        *,
        text: Optional[str] = None,
        reference_wav: Optional[str] = None,
        target_wav: Optional[str] = None,  # for VC: content source
        seed: Optional[int] = None,
    ) -> AttackResult:
        """Produce one spoofed sample.

        Different generators use different fields:
          - TTS:        requires text and reference_wav (for voice clone)
          - VC:         requires reference_wav (target speaker) and target_wav (content)
        """


class PostProcessor(ABC):
    """A waveform-in / waveform-out transform applied AFTER generation."""

    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def apply(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray: ...


class DummyTTS(AttackGenerator):
    """Always-available placeholder. Generates a noise burst — for plumbing tests."""

    name = "dummy_tts"
    family = "tts"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        *,
        text: Optional[str] = None,
        reference_wav: Optional[str] = None,
        target_wav: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> AttackResult:
        rng = np.random.default_rng(seed)
        n = 16000 * 3  # 3 sec at 16k
        wav = (rng.standard_normal(n) * 0.05).astype(np.float32)
        return AttackResult(
            waveform=wav,
            sample_rate=16000,
            metadata=AttackMetadata(
                algorithm=self.name,
                family=self.family,
                reference_wav=reference_wav,
                text=text,
            ),
        )


def list_available(generators: Sequence[AttackGenerator]) -> list[str]:
    return [g.name for g in generators if g.is_available()]
