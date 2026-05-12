"""TTS-based attack generators (zero-shot voice cloning)."""

from .xtts import XTTSAttack
from .openvoice import OpenVoiceAttack
from .tortoise import TortoiseAttack
from .bark import BarkAttack

ALL_TTS = [XTTSAttack, OpenVoiceAttack, TortoiseAttack, BarkAttack]

__all__ = ["XTTSAttack", "OpenVoiceAttack", "TortoiseAttack", "BarkAttack", "ALL_TTS"]
