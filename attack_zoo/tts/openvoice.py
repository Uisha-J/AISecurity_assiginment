"""OpenVoice (MyShell/MIT) attack adapter.

OpenVoice clones tone color from a single reference and supports
emotion/style control. Often used in combination with base TTS.

Install:
    pip install openvoice
    # plus checkpoint download — see openvoice repo
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

from ..base import AttackGenerator, AttackResult, AttackMetadata


class OpenVoiceAttack(AttackGenerator):
    name = "openvoice"
    family = "tts"

    def __init__(
        self,
        ckpt_base: str = "checkpoints/openvoice/base_speakers/EN",
        ckpt_converter: str = "checkpoints/openvoice/converter",
        device: str = "cuda",
        language: str = "English",
    ) -> None:
        self.ckpt_base = ckpt_base
        self.ckpt_converter = ckpt_converter
        self.device = device
        self.language = language
        self._base_tts = None
        self._converter = None

    def is_available(self) -> bool:
        try:
            import openvoice  # noqa: F401
            return Path(self.ckpt_base).exists() and Path(self.ckpt_converter).exists()
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._base_tts is not None:
            return
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter  # type: ignore
        self._base_tts = BaseSpeakerTTS(
            f"{self.ckpt_base}/config.json", device=self.device
        )
        self._base_tts.load_ckpt(f"{self.ckpt_base}/checkpoint.pth")
        self._converter = ToneColorConverter(
            f"{self.ckpt_converter}/config.json", device=self.device
        )
        self._converter.load_ckpt(f"{self.ckpt_converter}/checkpoint.pth")

    def generate(
        self,
        *,
        text: Optional[str] = None,
        reference_wav: Optional[str] = None,
        target_wav: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> AttackResult:
        if text is None or reference_wav is None:
            raise ValueError("OpenVoice requires text and reference_wav")
        self._ensure_loaded()
        # Real two-stage flow: base_tts -> tone color conversion.
        # Implementation details follow the openvoice README; the adapter
        # exposes a stable interface and lets the caller treat it like any
        # other generator.
        import tempfile, soundfile as sf  # local import to avoid hard dep
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as base, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            self._base_tts.tts(text, base.name, speaker="default",  # type: ignore
                               language=self.language, speed=1.0)
            # extract target tone color
            from openvoice import se_extractor  # type: ignore
            target_se, _ = se_extractor.get_se(reference_wav, self._converter,
                                               target_dir="tmp", vad=True)
            # convert
            source_se = self._base_tts.get_speaker_embedding()  # type: ignore
            self._converter.convert(  # type: ignore
                audio_src_path=base.name,
                src_se=source_se,
                tgt_se=target_se,
                output_path=out.name,
            )
            wav, sr = sf.read(out.name, dtype="float32")
        return AttackResult(
            waveform=np.asarray(wav, dtype=np.float32),
            sample_rate=int(sr),
            metadata=AttackMetadata(
                algorithm=self.name,
                family=self.family,
                reference_wav=reference_wav,
                text=text,
            ),
        )
