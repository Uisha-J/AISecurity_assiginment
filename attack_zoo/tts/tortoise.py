"""Tortoise-TTS attack adapter (neonbjb/tortoise-tts).

Slow but high-quality voice cloning. Held out from training in default
protocol — see protocols/unseen.yaml.

Install:
    pip install tortoise-tts
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from ..base import AttackGenerator, AttackResult, AttackMetadata


class TortoiseAttack(AttackGenerator):
    name = "tortoise"
    family = "tts"

    def __init__(self, preset: str = "fast", device: str = "cuda") -> None:
        self.preset = preset
        self.device = device
        self._tts = None

    def is_available(self) -> bool:
        try:
            import tortoise  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._tts is not None:
            return
        from tortoise.api import TextToSpeech  # type: ignore
        self._tts = TextToSpeech()

    def generate(
        self,
        *,
        text: Optional[str] = None,
        reference_wav: Optional[str] = None,
        target_wav: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> AttackResult:
        if text is None:
            raise ValueError("Tortoise requires text")
        self._ensure_loaded()
        assert self._tts is not None

        from tortoise.utils.audio import load_audio  # type: ignore
        voice_samples = [load_audio(reference_wav, 22050)] if reference_wav else None
        gen = self._tts.tts_with_preset(
            text=text,
            voice_samples=voice_samples,
            preset=self.preset,
        )
        wav = gen.squeeze().cpu().numpy().astype(np.float32)
        return AttackResult(
            waveform=wav,
            sample_rate=24000,
            metadata=AttackMetadata(
                algorithm=self.name,
                family=self.family,
                reference_wav=reference_wav,
                text=text,
            ),
        )
