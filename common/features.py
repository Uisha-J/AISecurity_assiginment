"""Feature extraction: LFCC, Mel-spectrogram."""

import numpy as np
import torch
import torchaudio


def extract_mel_spectrogram(waveform, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)(waveform)
    return torch.log(mel + 1e-9).squeeze(0).numpy()


def extract_lfcc(waveform, sr=16000, n_lfcc=60, n_fft=512, hop_length=160, n_filter=128):
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                       window=torch.hann_window(n_fft), return_complex=True)
    power = spec.abs().pow(2).squeeze(0)
    fb = torch.from_numpy(_linear_filterbank(n_filter, power.shape[0], sr, n_fft)).float()
    filtered = torch.log(torch.matmul(fb, power) + 1e-9)
    return _dct(filtered, n_lfcc).numpy()


def _linear_filterbank(n_filter, n_freq, sr, n_fft):
    centers = np.linspace(0, sr / 2, n_filter + 2)
    bins = np.floor((n_fft + 1) * centers / sr).astype(int)
    fb = np.zeros((n_filter, n_freq))
    for i in range(n_filter):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        for j in range(l, min(c, n_freq)):
            if c != l: fb[i, j] = (j - l) / (c - l)
        for j in range(c, min(r, n_freq)):
            if r != c: fb[i, j] = (r - j) / (r - c)
    return fb


def _dct(x, n_out):
    n_in = x.shape[0]
    k = torch.arange(n_out, dtype=x.dtype).unsqueeze(1)
    n = torch.arange(n_in, dtype=x.dtype).unsqueeze(0)
    return torch.matmul(torch.cos(np.pi * k * (2 * n + 1) / (2 * n_in)), x)
