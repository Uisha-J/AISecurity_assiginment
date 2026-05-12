"""End-to-end spoof detector composing frontend + backend + loss head."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .frontend import SSLFrontend
from .backend import AASIST
from .loss import OCSoftmaxLoss


@dataclass
class SpoofDetectorConfig:
    ssl_pretrained: str = "microsoft/wavlm-base-plus"
    ssl_freeze: bool = True
    ssl_weighted: bool = True
    gat_dim: int = 64
    n_subgraph_nodes: int = 32
    embed_dim: int = 128
    dropout: float = 0.1
    sample_rate: int = 16000

    # loss
    r_real: float = 0.9
    r_fake: float = 0.2
    alpha: float = 20.0


class SpoofDetector(nn.Module):
    def __init__(self, cfg: SpoofDetectorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frontend = SSLFrontend(
            pretrained=cfg.ssl_pretrained,
            freeze=cfg.ssl_freeze,
            output_layer_weighted=cfg.ssl_weighted,
            sample_rate=cfg.sample_rate,
        )
        self.backend = AASIST(
            in_dim=self.frontend.hidden_size,
            gat_dim=cfg.gat_dim,
            n_subgraph_nodes=cfg.n_subgraph_nodes,
            embed_dim=cfg.embed_dim,
            dropout=cfg.dropout,
        )
        self.loss_head = OCSoftmaxLoss(
            feat_dim=cfg.embed_dim,
            r_real=cfg.r_real,
            r_fake=cfg.r_fake,
            alpha=cfg.alpha,
        )

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        feats = self.frontend(waveform)           # (B, T, D)
        emb = self.backend(feats)                 # (B, E)
        return emb

    def score(self, waveform: torch.Tensor) -> torch.Tensor:
        """Bonafide-ness score: high == genuine."""
        emb = self.embed(waveform)
        return self.loss_head.score(emb)

    def forward(
        self, waveform: torch.Tensor, labels: Optional[torch.Tensor] = None
    ):
        emb = self.embed(waveform)
        if labels is None:
            return self.loss_head.score(emb)
        loss, scores = self.loss_head(emb, labels)
        return loss, scores, emb
