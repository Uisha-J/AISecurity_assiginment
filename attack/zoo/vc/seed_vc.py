"""seed-vc (Plachtaa/seed-vc) zero-shot voice conversion adapter.

Held-out attack in default protocol (unseen.yaml).

Install: clone Plachtaa/seed-vc into PYTHONPATH; weights download on first run.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from ..base import AttackGenerator, AttackResult, AttackMetadata


class SeedVCAttack(AttackGenerator):
    name = "seed_vc"
    family = "vc"

    def __init__(self, device: str = "cuda", diffusion_steps: int = 25) -> None:
        self.device = device
        self.diffusion_steps = diffusion_steps
        self._vc = None

    def is_available(self) -> bool:
        try:
            import seed_vc  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._vc is not None:
            return
        from seed_vc import SeedVC  # type: ignore
        self._vc = SeedVC(device=self.device)

    def generate(
        self,
        *,
        text: Optional[str] = None,
        reference_wav: Optional[str] = None,
        target_wav: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> AttackResult:
        if reference_wav is None or target_wav is None:
            raise ValueError("seed-vc requires reference_wav (target speaker) and target_wav (content)")
        self._ensure_loaded()
        assert self._vc is not None
        wav, sr = self._vc.convert(
            source=target_wav, target=reference_wav,
            diffusion_steps=self.diffusion_steps,
        )
        return AttackResult(
            waveform=np.asarray(wav, dtype=np.float32),
            sample_rate=int(sr),
            metadata=AttackMetadata(
                algorithm=self.name,
                family=self.family,
                reference_wav=reference_wav,
                text=None,
                extra={"content_wav": target_wav,
                       "diffusion_steps": self.diffusion_steps},
            ),
        )
