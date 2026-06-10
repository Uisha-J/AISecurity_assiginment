"""RVC (Retrieval-based Voice Conversion) attack adapter.

RVC is the dominant real-time voice changer used for voice-phishing
scenarios. We invoke it offline here for spoof generation.

Install (manual): clone RVC-Project/Retrieval-based-Voice-Conversion-WebUI
and add to PYTHONPATH, or use the lighter rvc-python wrapper.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

from ..base import AttackGenerator, AttackResult, AttackMetadata


class RVCAttack(AttackGenerator):
    name = "rvc"
    family = "vc"

    def __init__(
        self,
        model_pth: str,           # speaker conversion model (.pth)
        index_path: Optional[str] = None,
        device: str = "cuda",
        f0_method: str = "rmvpe",
        pitch_shift: int = 0,
    ) -> None:
        self.model_pth = model_pth
        self.index_path = index_path
        self.device = device
        self.f0_method = f0_method
        self.pitch_shift = pitch_shift
        self._vc = None

    def is_available(self) -> bool:
        try:
            import rvc_python  # noqa: F401
        except ImportError:
            return False
        return Path(self.model_pth).exists()

    def _ensure_loaded(self) -> None:
        if self._vc is not None:
            return
        from rvc_python.infer import RVCInference  # type: ignore
        self._vc = RVCInference(device=self.device)
        self._vc.load_model(self.model_pth, index_path=self.index_path)

    def generate(
        self,
        *,
        text: Optional[str] = None,
        reference_wav: Optional[str] = None,   # speaker (encoded in model_pth)
        target_wav: Optional[str] = None,       # CONTENT source audio
        seed: Optional[int] = None,
    ) -> AttackResult:
        if target_wav is None:
            raise ValueError("RVC requires `target_wav` (content audio to convert)")
        self._ensure_loaded()
        assert self._vc is not None

        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            self._vc.infer_file(
                input_path=target_wav,
                output_path=out.name,
                f0_method=self.f0_method,
                f0_up_key=self.pitch_shift,
            )
            wav, sr = sf.read(out.name, dtype="float32")
        return AttackResult(
            waveform=np.asarray(wav, dtype=np.float32),
            sample_rate=int(sr),
            metadata=AttackMetadata(
                algorithm=self.name,
                family=self.family,
                reference_wav=self.model_pth,
                text=None,
                extra={
                    "content_wav": target_wav,
                    "f0_method": self.f0_method,
                    "pitch_shift": self.pitch_shift,
                },
            ),
        )
