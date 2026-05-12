"""SSL frontend: WavLM / wav2vec2 / XLS-R wrapped for the spoof detector.

We extract per-layer hidden states and learn a small softmax weight over
layers (Hubert-style "weighted sum"). The frontend is kept FROZEN by default
because (a) it generalizes better and (b) it saves 99% of training compute.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class SSLFrontend(nn.Module):
    """Wraps a HuggingFace SSL model. Output: (B, T', D)."""

    def __init__(
        self,
        pretrained: str = "microsoft/wavlm-base-plus",
        freeze: bool = True,
        output_layer_weighted: bool = True,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoFeatureExtractor
        except ImportError as e:
            raise ImportError(
                "transformers required for SSLFrontend. "
                "pip install transformers"
            ) from e
        self.pretrained = pretrained
        self.sample_rate = sample_rate
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained)
        self.model = AutoModel.from_pretrained(pretrained, output_hidden_states=True)
        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        self.use_weighted = output_layer_weighted
        # +1 for the input embedding layer
        n_layers = self.model.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def train(self, mode: bool = True):  # keep frozen model in eval
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, T) float32 in [-1, 1] at self.sample_rate.
        returns:  (B, T', D)
        """
        # HF models accept raw float waveform tensors; no need to renormalize.
        if self.freeze:
            with torch.no_grad():
                out = self.model(waveform, output_hidden_states=True)
        else:
            out = self.model(waveform, output_hidden_states=True)

        if self.use_weighted:
            # (n_layers, B, T, D)
            hs = torch.stack(out.hidden_states, dim=0)
            w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            feats = (hs * w).sum(dim=0)
        else:
            feats = out.last_hidden_state
        return feats
