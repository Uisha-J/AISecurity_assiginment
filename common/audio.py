"""Audio I/O and processing utilities.

Loading uses soundfile + librosa (not torchaudio.load) so it works without
torchaudio's codec backend (torchcodec), which is not always installed.
"""

from pathlib import Path
import numpy as np
import librosa
import soundfile as sf


def load_audio(path, target_sr=16000, mono=True):
    waveform, sr = sf.read(str(path), dtype="float32")
    if mono and waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    return waveform.astype(np.float32), target_sr


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
