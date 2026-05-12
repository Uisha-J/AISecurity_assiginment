"""Data pipeline: protocol-driven dataset + on-the-fly augmentation."""

from .dataset import ProtocolDataset, ProtocolSpec, load_protocol
from .augment import build_augmentation_chain, RawBoost

__all__ = [
    "ProtocolDataset",
    "ProtocolSpec",
    "load_protocol",
    "build_augmentation_chain",
    "RawBoost",
]
