"""One-Class Softmax loss for spoof detection.

Reference: Zhang et al., "One-class Learning Towards Synthetic Voice Spoofing
Detection" (2021).

Key idea: pull genuine (bonafide) embeddings into a tight cluster around a
learned center w (angle < theta_real), and push spoof embeddings AWAY from
the center (angle > theta_fake). Unseen spoof types — regardless of where
their distribution sits — get rejected as long as they're not near the
single bonafide center.

This is the workhorse loss for unknown-attack defense.
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OCSoftmaxLoss(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        r_real: float = 0.9,
        r_fake: float = 0.2,
        alpha: float = 20.0,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, feat_dim))
        nn.init.kaiming_uniform_(self.center, a=math.sqrt(5))

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Higher score == more likely bonafide."""
        w = F.normalize(self.center, dim=1)
        e = F.normalize(embeddings, dim=1)
        return (e * w).sum(dim=1)                       # cosine sim, (B,)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        embeddings: (B, D)
        labels:     (B,) with 1 == bonafide, 0 == spoof
        Returns: (loss, score)
        """
        cos = self.score(embeddings)                    # (B,)
        # margin: target - cos for bonafide; cos - target for spoof
        m = torch.where(
            labels.bool(),
            self.r_real - cos,                           # want cos >= r_real
            cos - self.r_fake,                           # want cos <= r_fake
        )
        loss = F.softplus(self.alpha * m).mean()
        return loss, cos
