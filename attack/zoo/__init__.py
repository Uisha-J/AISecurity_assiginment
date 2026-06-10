"""Attack zoo: Blue-team self-attack generators.

The defender uses these to synthesize spoofed audio for training and held-out
evaluation. Splits are enforced at the algorithm level — see protocols/*.yaml.

Each adapter follows the AttackGenerator protocol defined in base.py.
Heavy ML dependencies (transformers, TTS, openvoice, ...) are imported lazily
so the rest of the pipeline runs without them.
"""

from .base import AttackGenerator, AttackResult, AttackMetadata
from . import tts, vc, post_process

__all__ = [
    "AttackGenerator",
    "AttackResult",
    "AttackMetadata",
    "tts",
    "vc",
    "post_process",
]
