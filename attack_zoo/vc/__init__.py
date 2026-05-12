"""Voice conversion attack generators (real-time / streaming family)."""

from .rvc import RVCAttack
from .seed_vc import SeedVCAttack

ALL_VC = [RVCAttack, SeedVCAttack]

__all__ = ["RVCAttack", "SeedVCAttack", "ALL_VC"]
