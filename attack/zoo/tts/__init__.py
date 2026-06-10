"""TTS-based attack generators (zero-shot voice cloning)."""

from .synthetic_tts import SyntheticTTSAttack
from .xtts import XTTSAttack
from .openvoice import OpenVoiceAttack
from .tortoise import TortoiseAttack
from .bark import BarkAttack

ALL_TTS = [SyntheticTTSAttack, XTTSAttack, OpenVoiceAttack, TortoiseAttack, BarkAttack]

__all__ = [
    "SyntheticTTSAttack",
    "XTTSAttack",
    "OpenVoiceAttack",
    "TortoiseAttack",
    "BarkAttack",
    "ALL_TTS",
]
