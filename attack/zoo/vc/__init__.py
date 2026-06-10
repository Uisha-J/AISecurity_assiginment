"""Voice conversion attack generators (real-time / streaming family)."""

from .artifact_vc import ArtifactVCAttack
from .rvc import RVCAttack
from .seed_vc import SeedVCAttack

ALL_VC = [ArtifactVCAttack, RVCAttack, SeedVCAttack]

__all__ = ["ArtifactVCAttack", "RVCAttack", "SeedVCAttack", "ALL_VC"]
