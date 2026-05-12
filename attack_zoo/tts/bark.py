"""Suno Bark attack adapter.

Bark generates speech with non-verbal sounds (laughs, sighs).
Used as a HELD-OUT attack in default protocol — see protocols/unseen.yaml.

Install:
    pip install git+https://github.com/suno-ai/bark.git
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from ..base import AttackGenerator, AttackResult, AttackMetadata


class BarkAttack(AttackGenerator):
    name = "bark"
    family = "tts"

    def __init__(self, voice_preset: Optional[str] = None) -> None:
        self.voice_preset = voice_preset  # e.g. "v2/en_speaker_6"
        self._loaded = False

    def is_available(self) -> bool:
        try:
            import bark  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from bark import preload_models  # type: ignore
        preload_models()
        self._loaded = True

    def generate(
        self,
        *,
        text: Optional[str] = None,
        reference_wav: Optional[str] = None,
        target_wav: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> AttackResult:
        if text is None:
            raise ValueError("Bark requires text")
        self._ensure_loaded()
        from bark import SAMPLE_RATE, generate_audio  # type: ignore
        wav = generate_audio(text, history_prompt=self.voice_preset)
        wav = np.asarray(wav, dtype=np.float32)
        return AttackResult(
            waveform=wav,
            sample_rate=int(SAMPLE_RATE),
            metadata=AttackMetadata(
                algorithm=self.name,
                family=self.family,
                reference_wav=reference_wav,
                text=text,
                extra={"voice_preset": self.voice_preset},
            ),
        )
