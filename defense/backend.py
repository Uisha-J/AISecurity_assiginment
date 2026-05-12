"""AASIST-style backend.

Faithful-in-spirit implementation: a graph-attention module over time- and
frequency-axis (here: SSL feature axis) sub-graphs, with heterogeneous
fusion, ending in a pooled embedding.

Reference: Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
Spectro-Temporal Graph Attention Networks", ICASSP 2022.

This is a compact reimplementation tuned to ride on top of an SSL frontend
(input shape (B, T, D)) rather than raw waveform / SincNet, which is what
the SSL+AASIST line of work uses.
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Single-head GAT layer with learnable attention over node pairs."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: (B, N, F)
        Wh = self.W(h)                                # (B, N, out)
        e = self.a_src(Wh) + self.a_dst(Wh).transpose(1, 2)  # (B, N, N)
        e = self.act(e)
        if mask is not None:
            e = e.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        return alpha @ Wh                              # (B, N, out)


class HeterogeneousAttention(nn.Module):
    """Cross-attention between two graphs (temporal <-> spectral)."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = math.sqrt(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        # x_a attends to x_b
        q = self.q(x_a)
        k = self.k(x_b)
        v = self.v(x_b)
        attn = q @ k.transpose(1, 2) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return attn @ v


class AASIST(nn.Module):
    """SSL-friendly AASIST backend.

    Input:  (B, T, D)  — token sequence from an SSL frontend
    Output: (B, embed_dim) — sample-level utterance embedding
    """

    def __init__(
        self,
        in_dim: int,
        gat_dim: int = 64,
        n_subgraph_nodes: int = 32,   # downsample T -> N nodes
        embed_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, gat_dim)
        self.n_nodes = n_subgraph_nodes

        # Two graphs from two different views of the feature map:
        #   - "temporal" graph: nodes pool along time
        #   - "spectral" graph: nodes pool along feature dim
        self.gat_t = GraphAttentionLayer(gat_dim, gat_dim, dropout)
        self.gat_s = GraphAttentionLayer(gat_dim, gat_dim, dropout)
        self.hetero_ts = HeterogeneousAttention(gat_dim, dropout)
        self.hetero_st = HeterogeneousAttention(gat_dim, dropout)

        self.head = nn.Sequential(
            nn.Linear(gat_dim * 4, gat_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gat_dim * 2, embed_dim),
        )

    @staticmethod
    def _pool_time(x: torch.Tensor, n_nodes: int, mode: str) -> torch.Tensor:
        """Pool the time axis of (B, T, D) down to n_nodes.

        mode == "avg": adaptive average pool
        mode == "max": adaptive max pool
        Returns (B, n_nodes, D).
        """
        x = x.transpose(1, 2)                          # (B, D, T)
        if mode == "max":
            x = F.adaptive_max_pool1d(x, n_nodes)
        else:
            x = F.adaptive_avg_pool1d(x, n_nodes)
        return x.transpose(1, 2)                       # (B, n_nodes, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.proj(x)                               # (B, T, gat_dim)

        # Two views of the same sequence: smooth (avg) vs salient (max).
        # Both stay (B, n_nodes, gat_dim) so the cross-graph attention shapes
        # line up. This is the SSL-friendly stand-in for AASIST's original
        # spectro-temporal dual view (we have no frequency axis here).
        nodes_t = self._pool_time(x, self.n_nodes, mode="avg")
        nodes_s = self._pool_time(x, self.n_nodes, mode="max")

        h_t = self.gat_t(nodes_t)
        h_s = self.gat_s(nodes_s)
        h_ts = self.hetero_ts(h_t, h_s)
        h_st = self.hetero_st(h_s, h_t)

        # Aggregate: max + mean over nodes from each branch, concat
        def agg(h):
            return torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)

        z = torch.cat([agg(h_t + h_ts), agg(h_s + h_st)], dim=-1)
        return self.head(z)                            # (B, embed_dim)
