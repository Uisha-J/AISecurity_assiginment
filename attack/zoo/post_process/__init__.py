"""Post-processing modules applied after spoof generation.

Critical: most defenders fail not because of new attack models but because
attackers route audio through codecs and noisy channels. Post-processing
diversity matters more than model diversity.
"""

from .codec import CodecPostProcessor
from .room_ir import RoomIRPostProcessor
from .noise_mix import NoisePostProcessor

ALL_POST = [CodecPostProcessor, RoomIRPostProcessor, NoisePostProcessor]

__all__ = ["CodecPostProcessor", "RoomIRPostProcessor", "NoisePostProcessor", "ALL_POST"]
