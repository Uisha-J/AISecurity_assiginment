"""Datasets for attack/defense experiments: LibriSpeech, ASVspoof, ClonedVoice."""

from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio

from .features import extract_lfcc
from .audio import load_audio, save_audio, concatenate_utterances


class LibriSpeechSpeakerDataset:
    def __init__(self, data_root, subset="test-clean"):
        self.data_root = Path(data_root) / "librispeech"
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=str(self.data_root), url=subset, download=False)
        self.speaker_utterances = defaultdict(list)
        for i in range(len(self.dataset)):
            _, _, _, spk, ch, utt = self.dataset[i]
            self.speaker_utterances[spk].append(
                str(self.data_root / "LibriSpeech" / subset / str(spk) / str(ch) / f"{spk}-{ch}-{utt:04d}.flac"))

    def get_utterances(self, spk_id):
        return self.speaker_utterances[spk_id]

    def select_speakers(self, n):
        return sorted(self.speaker_utterances, key=lambda s: len(self.speaker_utterances[s]), reverse=True)[:n]


class ASVspoofDataset(Dataset):
    """ASVspoof2019 LA dataset.

    Label convention (project-wide): 1 == bonafide, 0 == spoof.
    """
    def __init__(self, data_root, split="train", feature_type="lfcc", max_audio_len=64000, n_lfcc=60):
        self.feature_type, self.max_audio_len, self.n_lfcc = feature_type, max_audio_len, n_lfcc
        root = Path(data_root)
        split_map = {"train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
                     "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
                     "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt")}
        adir, pfile = split_map[split]
        self.audio_dir = root / adir / "flac"
        self.samples = []
        with open(root / "ASVspoof2019_LA_cm_protocols" / pfile) as f:
            for line in f:
                p = line.strip().split()
                self.samples.append({"audio_path": str(self.audio_dir / f"{p[1]}.flac"), "label": 1 if p[4] == "bonafide" else 0})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav, _ = load_audio(s["audio_path"], target_sr=16000)
        wav = wav[:self.max_audio_len]
        if self.feature_type == "raw":
            if len(wav) < self.max_audio_len: wav = np.pad(wav, (0, self.max_audio_len - len(wav)))
            return torch.from_numpy(wav).float(), s["label"]
        return torch.from_numpy(extract_lfcc(wav, n_lfcc=self.n_lfcc)).float(), s["label"]

    def get_class_weights(self):
        labels = [s["label"] for s in self.samples]
        n0, n1, t = labels.count(0), labels.count(1), len(labels)
        return torch.tensor([t / (2 * n0), t / (2 * n1)])


class ClonedVoiceDataset(Dataset):
    """Real vs. cloned-voice pairs.

    Label convention (project-wide): 1 == bonafide (real), 0 == spoof (cloned).
    """
    def __init__(self, real_paths, cloned_paths, feature_type="lfcc", max_audio_len=64000, n_lfcc=60):
        self.feature_type, self.max_audio_len, self.n_lfcc = feature_type, max_audio_len, n_lfcc
        self.samples = [{"audio_path": p, "label": 1} for p in real_paths] + [{"audio_path": p, "label": 0} for p in cloned_paths]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav, _ = load_audio(s["audio_path"], target_sr=16000)
        wav = wav[:self.max_audio_len]
        if self.feature_type == "raw":
            if len(wav) < self.max_audio_len: wav = np.pad(wav, (0, self.max_audio_len - len(wav)))
            return torch.from_numpy(wav).float(), s["label"]
        return torch.from_numpy(extract_lfcc(wav, n_lfcc=self.n_lfcc)).float(), s["label"]


def prepare_reference_samples(utterance_paths, output_dir, speaker_id, durations=None, sr=16000):
    if durations is None: durations = [5, 10, 30]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for d in durations:
        out = output_dir / f"ref_{speaker_id}_{d}s.wav"
        audio = concatenate_utterances(utterance_paths, target_duration=d, sr=sr)
        mx = np.abs(audio).max()
        if mx > 0: audio = audio / mx
        save_audio(audio, str(out), sr=sr)
        results[d] = str(out)
    return results
