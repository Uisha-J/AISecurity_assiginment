"""Defense model: SSL frontend + AASIST backend + OC-Softmax loss."""

from .frontend import SSLFrontend
from .backend import AASIST
from .loss import OCSoftmaxLoss
from .model import SpoofDetector

__all__ = ["SSLFrontend", "AASIST", "OCSoftmaxLoss", "SpoofDetector"]
