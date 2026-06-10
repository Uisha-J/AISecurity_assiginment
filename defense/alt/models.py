"""Alternative detector architectures: LCNN and RawNet2.

LCNN operates on LFCC features; RawNet2 on raw waveforms via SincNet.
Based on ASVspoof 2021 baselines.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ LCNN

class MaxFeatureMap(nn.Module):
    def forward(self, x):
        out1, out2 = x.chunk(2, dim=1)
        return torch.max(out1, out2)


class LCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.mfm = MaxFeatureMap()

    def forward(self, x):
        return self.mfm(self.bn(self.conv(x)))


class LCNN(nn.Module):
    """Light CNN for anti-spoofing. Input: (B, n_lfcc, T) -> (B, 2)."""
    def __init__(self, n_lfcc=60):
        super().__init__()
        self.features = nn.Sequential(
            LCNNBlock(1, 32, 5, 2), nn.MaxPool2d(2, 2),
            LCNNBlock(32, 32, 1, 0), LCNNBlock(32, 48, 3, 1), nn.MaxPool2d(2, 2),
            LCNNBlock(48, 48, 1, 0), LCNNBlock(48, 64, 3, 1), nn.MaxPool2d(2, 2),
            LCNNBlock(64, 64, 1, 0), LCNNBlock(64, 32, 3, 1),
            LCNNBlock(32, 32, 1, 0), LCNNBlock(32, 32, 3, 1), nn.MaxPool2d(2, 2),
        )
        # Adaptive pooling makes the head independent of input length AND n_lfcc,
        # so the model accepts variable-duration audio without shape mismatch.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(self.features(x))
        return self.fc(x.view(x.size(0), -1))


# ------------------------------------------------------------------ RawNet2

class SincConv(nn.Module):
    def __init__(self, out_channels=128, kernel_size=1024, sample_rate=16000,
                 min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels, self.kernel_size = out_channels, kernel_size
        self.sample_rate, self.min_low_hz, self.min_band_hz = sample_rate, min_low_hz, min_band_hz
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel_pts = torch.linspace(2595 * math.log10(1 + min_low_hz / 700),
                                 2595 * math.log10(1 + high_hz / 700), out_channels + 1)
        hz = 700 * (10 ** (mel_pts / 2595) - 1)
        self.low_hz_ = nn.Parameter(hz[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz[1:] - hz[:-1]).unsqueeze(1))
        n = (kernel_size - 1) / 2.0
        self.register_buffer("window", 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(-n, 0) / kernel_size))
        self.register_buffer("n_", 2 * math.pi * torch.arange(-n, 0).unsqueeze(0) / sample_rate)

    def forward(self, x):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        f_l, f_h = torch.matmul(low, self.n_), torch.matmul(high, self.n_)
        left = (torch.sin(f_h) - torch.sin(f_l)) / (self.n_ / 2) * self.window
        center = 2 * (high - low)
        bp = torch.cat([left, center, torch.flip(left, [1])], dim=1) / (2 * center)
        return F.conv1d(x, bp.unsqueeze(1), padding=self.kernel_size // 2)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn1, self.conv1 = nn.BatchNorm1d(in_ch), nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.bn2, self.conv2 = nn.BatchNorm1d(out_ch), nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.ds = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.pool, self.fms = nn.MaxPool1d(3), nn.Sigmoid()

    def forward(self, x):
        r = x
        o = self.conv1(F.leaky_relu(self.bn1(x), 0.3))
        o = self.conv2(F.leaky_relu(self.bn2(o), 0.3))
        if self.ds:
            r = self.ds(r)
        o = (o + r) * self.fms((o + r).mean(dim=-1, keepdim=True))
        return self.pool(o)


class RawNet2(nn.Module):
    """RawNet2 end-to-end anti-spoofing. Input: (B, n_samples) -> (B, 2)."""
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sinc = SincConv(128, 1024, sample_rate)
        self.bn_sinc, self.pool_sinc = nn.BatchNorm1d(128), nn.MaxPool1d(3)
        self.res = nn.Sequential(ResBlock(128, 128), ResBlock(128, 256),
                                 ResBlock(256, 256), ResBlock(256, 256),
                                 ResBlock(256, 256), ResBlock(256, 256))
        self.bn_gru = nn.BatchNorm1d(256)
        self.gru = nn.GRU(256, 1024, 3, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(1024, 2)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool_sinc(F.leaky_relu(self.bn_sinc(self.sinc(x)), 0.3))
        x = self.bn_gru(self.res(x)).permute(0, 2, 1)
        _, h = self.gru(x)
        return self.fc(h[-1])
