"""Protocol-driven PyTorch Dataset.

A "protocol" is a YAML file that lists which directories contain bonafide
and spoofed audio (with attack tags). The dataset loads files, resamples,
crops to a fixed length, optionally augments, and returns
(waveform, label, attack_tag).

This is the single source of truth for splits: NEVER bypass the protocol.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


@dataclass
class ProtocolEntry:
    path: Path
    label: int                       # 1 == bonafide, 0 == spoof
    attack_tag: str = "bonafide"


@dataclass
class ProtocolSpec:
    name: str
    description: str
    entries: list[ProtocolEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def class_counts(self) -> dict:
        from collections import Counter
        return dict(Counter(e.attack_tag for e in self.entries))


def _resolve_glob(spec: dict, base_dir: Path) -> list[Path]:
    p = (base_dir / spec["path"]).resolve()
    glob = spec.get("glob", "**/*.wav")
    if not p.exists():
        return []
    return sorted(p.rglob(glob.replace("**/", "")))


def load_protocol(yaml_path: str | Path, base_dir: str | Path = ".") -> ProtocolSpec:
    yaml_path = Path(yaml_path)
    base = Path(base_dir).resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    entries: list[ProtocolEntry] = []
    for src in cfg.get("bonafide", []):
        for p in _resolve_glob(src, base):
            entries.append(ProtocolEntry(p, 1, "bonafide"))
    for src in cfg.get("spoof", []):
        tag = src.get("attack_tag", "spoof")
        for p in _resolve_glob(src, base):
            entries.append(ProtocolEntry(p, 0, tag))

    return ProtocolSpec(
        name=cfg.get("name", yaml_path.stem),
        description=cfg.get("description", ""),
        entries=entries,
    )


class ProtocolDataset(Dataset):
    def __init__(
        self,
        protocol: ProtocolSpec,
        sample_rate: int = 16000,
        segment_seconds: float = 4.0,
        train: bool = True,
        augment: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.protocol = protocol
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_seconds)
        self.train = train
        self.augment = augment

        if not protocol.entries:
            raise ValueError(
                f"Protocol '{protocol.name}' is empty; check paths in YAML."
            )

    def __len__(self) -> int:
        return len(self.protocol.entries)

    def _load(self, path: Path) -> np.ndarray:
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        return wav.astype(np.float32)

    def _crop_or_pad(self, wav: np.ndarray) -> np.ndarray:
        n = self.segment_samples
        if len(wav) >= n:
            if self.train:
                start = random.randint(0, len(wav) - n)
            else:
                start = (len(wav) - n) // 2
            return wav[start : start + n]
        # pad: cycle-pad rather than zero-pad to keep voicing
        reps = int(np.ceil(n / max(len(wav), 1)))
        return np.tile(wav, reps)[:n]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        entry = self.protocol.entries[idx]
        try:
            wav = self._load(entry.path)
        except Exception:
            # silent file in case of I/O hiccup
            wav = np.zeros(self.segment_samples, dtype=np.float32)
        wav = self._crop_or_pad(wav)
        if self.train and self.augment is not None:
            wav = self.augment(wav, self.sample_rate)
        return torch.from_numpy(wav), entry.label, entry.attack_tag
