"""Coqui XTTS v2 attack adapter.

XTTS v2 clones a speaker from ~6 sec of reference audio, generates speech
in 17 languages. The most common open-source weapon for sample-cloning
attacks today.

Install:
    pip install TTS
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from ..base import AttackGenerator, AttackResult, AttackMetadata


class XTTSAttack(AttackGenerator):
    name = "xtts_v2"
    family = "tts"

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        language: str = "en",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device
        self._tts = None  # lazy

    def is_available(self) -> bool:
        try:
            import TTS  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._tts is not None:
            return
        from TTS.api import TTS  # type: ignore
        self._tts = TTS(self.model_name).to(self.device)

    def generate(
        self,
        *,
        text: Optional[str] = None,
        reference_wav: Optional[str] = None,
        target_wav: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> AttackResult:
        if text is None:
            raise ValueError("XTTS requires `text`")
        if reference_wav is None:
            raise ValueError("XTTS requires `reference_wav` (speaker sample)")
        self._ensure_loaded()
        assert self._tts is not None

        # XTTS returns a list[float] at 24kHz
        wav = self._tts.tts(
            text=text,
            speaker_wav=reference_wav,
            language=self.language,
        )
        wav = np.asarray(wav, dtype=np.float32)
        return AttackResult(
            waveform=wav,
            sample_rate=24000,
            metadata=AttackMetadata(
                algorithm=self.name,
                family=self.family,
                reference_wav=reference_wav,
                text=text,
                extra={"language": self.language},
            ),
        )
