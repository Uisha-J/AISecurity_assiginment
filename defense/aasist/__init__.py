"""AASIST-based spoof detector: SSL frontend + graph-attention backend + OC-Softmax."""

from .frontend import SSLFrontend
from .backend import AASIST
from .loss import OCSoftmaxLoss
from .model import SpoofDetector

__all__ = ["SSLFrontend", "AASIST", "OCSoftmaxLoss", "SpoofDetector"]
