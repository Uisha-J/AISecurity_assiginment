"""Audio I/O and processing utilities."""

from pathlib import Path
import numpy as np
import torchaudio
import librosa
import soundfile as sf


def load_audio(path, target_sr=16000, mono=True):
    waveform, sr = torchaudio.load(path)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform.squeeze(0).numpy(), target_sr


def save_audio(waveform, path, sr=16000):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sr)


def trim_silence(waveform, sr=16000, top_db=30):
    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed


def concatenate_utterances(audio_paths, target_duration, sr=16000):
    segments, total, target = [], 0, int(target_duration * sr)
    for p in audio_paths:
        wav, _ = load_audio(p, target_sr=sr)
        segments.append(wav)
        total += len(wav)
        if total >= target:
            break
    return np.concatenate(segments)[:target]
