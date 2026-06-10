"""Red-team attack generation against the spoof detector.

This module builds controlled red-team submissions from authorized spoof wavs.
It does not clone a real person's voice; it mutates existing test samples and,
optionally, uses a defender checkpoint to select the variants most likely to
cross the detector threshold.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from ..common.augment import RawBoost


Transform = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class AttackCandidate:
    source_file: str
    output_file: str
    transform: str
    score: Optional[float] = None
    passed_threshold: Optional[bool] = None


@dataclass
class RedTeamAttackConfig:
    input_dir: str
    out_dir: str
    checkpoint: Optional[str] = None
    threshold: Optional[float] = None
    sample_rate: int = 16000
    segment_seconds: float = 4.0
    variants_per_file: int = 8
    keep_all: bool = False
    seed: int = 1337
    device: Optional[str] = None
    whitebox_steps: int = 0
    whitebox_epsilon: float = 0.002
    whitebox_step_size: float = 0.0005


def _safe_audio(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 1.0:
        x = x / peak
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _load_wav(path: Path, sr_target: int, segment_seconds: float) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sr_target:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_target)
    n = int(sr_target * segment_seconds)
    if len(wav) >= n:
        start = (len(wav) - n) // 2
        wav = wav[start : start + n]
    else:
        reps = int(np.ceil(n / max(len(wav), 1)))
        wav = np.tile(wav, reps)[:n]
    return _safe_audio(wav)


def _random_eq(x: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    from scipy import signal

    cutoff = rng.uniform(0.18, 0.46)
    taps = signal.firwin(rng.choice([9, 17, 31]), cutoff)
    y = signal.lfilter(taps, [1.0], x).astype(np.float32)
    mix = rng.uniform(0.35, 0.85)
    return _safe_audio((1.0 - mix) * x + mix * y)


def _micro_shift(x: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    shift = rng.randint(int(-0.015 * sr), int(0.015 * sr))
    if shift == 0:
        return x.copy()
    return _safe_audio(np.roll(x, shift))


def _gain_jitter(x: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    return _safe_audio(x * rng.uniform(0.65, 1.25))


def _gaussian_bed(x: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    snr_db = rng.uniform(18.0, 36.0)
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).standard_normal(len(x)).astype(np.float32)
    sig_p = float(np.mean(x**2)) + 1e-12
    noise = noise * np.sqrt((sig_p / (10 ** (snr_db / 10.0))) / (float(np.var(noise)) + 1e-12))
    return _safe_audio(x + noise)


def _resample_roundtrip(x: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    import librosa

    mid_sr = rng.choice([8000, 11025, 12000, 22050])
    y = librosa.resample(x, orig_sr=sr, target_sr=mid_sr)
    y = librosa.resample(y, orig_sr=mid_sr, target_sr=sr)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return _safe_audio(y[: len(x)])


def _build_transforms(rng: random.Random) -> list[tuple[str, Transform]]:
    rawboost = RawBoost(rng=rng)

    return [
        ("identity", lambda x, sr: x.copy()),
        ("gain_jitter", lambda x, sr: _gain_jitter(x, sr, rng)),
        ("micro_shift", lambda x, sr: _micro_shift(x, sr, rng)),
        ("random_eq", lambda x, sr: _random_eq(x, sr, rng)),
        ("gaussian_bed", lambda x, sr: _gaussian_bed(x, sr, rng)),
        ("resample_roundtrip", lambda x, sr: _resample_roundtrip(x, sr, rng)),
        ("rawboost", lambda x, sr: _safe_audio(rawboost(x, sr))),
    ]


class OptionalScorer:
    def __init__(
        self,
        checkpoint: Optional[str],
        threshold: Optional[float],
        device: Optional[str],
    ) -> None:
        self.model = None
        self.cfg: dict = {}
        self.threshold = threshold
        self.device = device
        if checkpoint is None:
            return

        import torch

        from .evaluate_submission import _build_model

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.cfg = _build_model(checkpoint, self.device)
        if self.threshold is None:
            ckpt = torch.load(checkpoint, map_location="cpu")
            self.threshold = float(ckpt.get("dev_threshold", 0.5))

    def available(self) -> bool:
        return self.model is not None

    def score_many(self, wavs: list[np.ndarray]) -> list[Optional[float]]:
        if self.model is None:
            return [None] * len(wavs)
        import torch

        x = torch.from_numpy(np.stack(wavs)).to(self.device)
        with torch.no_grad():
            scores = self.model.score(x).detach().cpu().numpy().tolist()
        return [float(s) for s in scores]

    def whitebox_pgd(
        self,
        wav: np.ndarray,
        *,
        steps: int,
        epsilon: float,
        step_size: float,
    ) -> np.ndarray:
        if self.model is None or steps <= 0:
            return wav.copy()
        import torch

        x0 = torch.from_numpy(wav).to(self.device).unsqueeze(0)
        x = x0.clone().detach()
        for _ in range(steps):
            x.requires_grad_(True)
            score = self.model.score(x).mean()
            grad = torch.autograd.grad(score, x)[0]
            x = x.detach() + step_size * grad.sign()
            x = torch.max(torch.min(x, x0 + epsilon), x0 - epsilon)
            x = torch.clamp(x, -1.0, 1.0)
        return x.squeeze(0).detach().cpu().numpy().astype(np.float32)


def generate_redteam_submission(cfg: RedTeamAttackConfig) -> dict:
    """Generate a red-team submission directory and return its manifest."""
    input_dir = Path(cfg.input_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    rng = random.Random(cfg.seed)
    transforms = _build_transforms(rng)
    scorer = OptionalScorer(cfg.checkpoint, cfg.threshold, cfg.device)

    candidates_out: list[AttackCandidate] = []
    labels_path = out_dir / "labels.csv"
    manifest_path = out_dir / "manifest.json"

    with labels_path.open("w", newline="", encoding="utf-8") as f_labels:
        labels_writer = csv.DictWriter(f_labels, fieldnames=["file", "label"])
        labels_writer.writeheader()

        for src_idx, wav_path in enumerate(wav_files):
            base = _load_wav(wav_path, cfg.sample_rate, cfg.segment_seconds)
            local: list[tuple[str, np.ndarray]] = []

            sampled = [rng.choice(transforms) for _ in range(max(1, cfg.variants_per_file))]
            for name, fn in sampled:
                local.append((name, fn(base, cfg.sample_rate)))
            if scorer.available() and cfg.whitebox_steps > 0:
                local.append((
                    "whitebox_pgd",
                    scorer.whitebox_pgd(
                        base,
                        steps=cfg.whitebox_steps,
                        epsilon=cfg.whitebox_epsilon,
                        step_size=cfg.whitebox_step_size,
                    ),
                ))

            scores = scorer.score_many([wav for _, wav in local])
            ranked = sorted(
                zip(local, scores),
                key=lambda item: float("-inf") if item[1] is None else item[1],
                reverse=True,
            )
            selected = ranked if cfg.keep_all else ranked[:1]

            for cand_idx, ((transform_name, wav), score) in enumerate(selected):
                suffix = cand_idx if cfg.keep_all else 0
                out_name = f"attack_{src_idx:06d}_{suffix:02d}_{transform_name}.wav"
                sf.write(str(out_dir / out_name), _safe_audio(wav), cfg.sample_rate)
                labels_writer.writerow({"file": out_name, "label": 0})
                passed = None
                if score is not None and scorer.threshold is not None:
                    passed = bool(score > scorer.threshold)
                candidates_out.append(AttackCandidate(
                    source_file=str(wav_path),
                    output_file=out_name,
                    transform=transform_name,
                    score=score,
                    passed_threshold=passed,
                ))

    scores_known = [c.score for c in candidates_out if c.score is not None]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "checkpoint": cfg.checkpoint,
        "threshold": scorer.threshold,
        "n_source_files": len(wav_files),
        "n_output_files": len(candidates_out),
        "keep_all": cfg.keep_all,
        "variants_per_file": cfg.variants_per_file,
        "whitebox_steps": cfg.whitebox_steps,
        "score_stats": None,
        "files": [asdict(c) for c in candidates_out],
    }
    if scores_known:
        arr = np.array(scores_known, dtype=np.float64)
        manifest["score_stats"] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_above_threshold": (
                int((arr > scorer.threshold).sum())
                if scorer.threshold is not None else None
            ),
        }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
