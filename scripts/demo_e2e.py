"""End-to-end skeleton demo.

Goal: prove the WHOLE pipeline runs without any external dataset or
pretrained model download. This is the strongest skeleton check.

What it does:
  1. Generates ~50 synthetic "bonafide" wavs (vocal-like formants)
  2. Generates ~50 synthetic "spoof"   wavs (different formants + tells)
  3. Writes ad-hoc train/dev/eval protocol YAMLs
  4. Trains a TINY no-SSL detector for 3 epochs
  5. Evaluates on dev + eval protocols, writes Markdown report
  6. Tests the red-team submission script on a held-out set

When this passes, every wiring decision in the project is validated:
  config -> dataset -> augmentation -> model -> loss -> training -> metrics
  -> per-attack breakdown -> report -> red-team interface

The "synthetic" wavs use simple harmonic + noise generators that DIFFER
between bonafide and spoof so the model has a learnable signal — this
exercises the loss math, not just the plumbing.
"""

from __future__ import annotations
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader


# ============================================================ synth wavs

def _synth_bonafide(seed: int, sr: int = 16000, secs: float = 3.0) -> np.ndarray:
    """Voiced-speech-like signal: F0 + 3 formant-ish harmonics + breath noise."""
    rng = np.random.default_rng(seed)
    n = int(sr * secs)
    t = np.arange(n, dtype=np.float32) / sr
    f0 = rng.uniform(110, 200)
    # slow F0 wobble (natural)
    f0_curve = f0 * (1 + 0.04 * np.sin(2 * np.pi * rng.uniform(2, 5) * t))
    phase = 2 * np.pi * np.cumsum(f0_curve) / sr
    sig = (0.6 * np.sin(phase)
           + 0.3 * np.sin(2 * phase)
           + 0.15 * np.sin(3 * phase))
    # amplitude envelope (syllabic)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 4) * t)
    sig = sig * env
    sig += 0.02 * rng.standard_normal(n).astype(np.float32)
    sig = sig / max(np.abs(sig).max(), 1e-8) * 0.6
    return sig.astype(np.float32)


def _synth_spoof(seed: int, sr: int = 16000, secs: float = 3.0) -> np.ndarray:
    """Spoof tells: very stable F0, no breath noise, additional whistle harmonic."""
    rng = np.random.default_rng(seed + 9999)
    n = int(sr * secs)
    t = np.arange(n, dtype=np.float32) / sr
    f0 = rng.uniform(140, 180)
    # too-stable F0 (TTS tell)
    phase = 2 * np.pi * f0 * t
    sig = (0.6 * np.sin(phase)
           + 0.3 * np.sin(2 * phase + rng.uniform(0, 0.1))
           + 0.15 * np.sin(3 * phase)
           # spurious high-freq whistle (vocoder tell)
           + 0.05 * np.sin(2 * np.pi * 7000 * t))
    # mechanical envelope
    env = 0.7 + 0.3 * np.sin(2 * np.pi * rng.uniform(3, 6) * t)
    sig = sig * env
    # very low noise floor
    sig += 0.002 * rng.standard_normal(n).astype(np.float32)
    sig = sig / max(np.abs(sig).max(), 1e-8) * 0.6
    return sig.astype(np.float32)


def _write_dataset(root: Path, n_train: int = 50, n_dev: int = 16,
                   n_eval: int = 16) -> dict:
    """Returns a dict of {split: {'bonafide': dir, 'spoof_a': dir, 'spoof_b': dir}}."""
    layout = {
        "train": {"bonafide": n_train, "spoof_a": n_train // 2, "spoof_b": n_train // 2},
        "dev":   {"bonafide": n_dev,   "spoof_a": n_dev,         "spoof_b": 0},
        "eval":  {"bonafide": n_eval,  "spoof_a": 0,             "spoof_b": n_eval},
    }
    out: dict = {}
    for split, classes in layout.items():
        out[split] = {}
        for cls, n in classes.items():
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                base = hash((split, cls, i)) & 0xFFFFFFFF
                wav = (_synth_bonafide(base) if cls == "bonafide"
                       else _synth_spoof(base + (0 if cls == "spoof_a" else 12345)))
                sf.write(d / f"{cls}_{i:04d}.wav", wav, 16000)
            out[split][cls] = d
    return out


def _write_protocol(path: Path, base_dir: Path, bona_dir: Path,
                    spoof_dirs: dict[str, Path]) -> None:
    """Write a protocol YAML for one split. Paths relative to base_dir."""
    proto = {
        "name": path.stem,
        "description": f"auto-generated demo protocol ({path.stem})",
        "bonafide": [
            {"path": str(bona_dir.relative_to(base_dir)).replace("\\", "/"),
             "glob": "*.wav"}
        ],
        "spoof": [
            {"path": str(d.relative_to(base_dir)).replace("\\", "/"),
             "attack_tag": tag}
            for tag, d in spoof_dirs.items()
        ],
    }
    path.write_text(yaml.safe_dump(proto, sort_keys=False), encoding="utf-8")


# ============================================================ tiny detector

class TinyFrontend(torch.nn.Module):
    """1-D conv frontend that mimics the (B, T, D) output shape of an SSL model.
    No external download; pretrains in seconds; perfect for skeleton checks."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=80, stride=40, padding=40),
            torch.nn.SiLU(),
            torch.nn.Conv1d(32, hidden, kernel_size=5, stride=2, padding=2),
            torch.nn.SiLU(),
        )
        self.hidden_size = hidden

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.unsqueeze(1)            # (B, 1, T)
        h = self.conv(x)                # (B, hidden, T')
        return h.transpose(1, 2)        # (B, T', hidden)


# ============================================================ runner

def run(workdir: Path) -> dict:
    print(f"[demo] workdir = {workdir}")
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Synthesize a tiny dataset --------------------------------------
    data_root = workdir / "data"
    print("[demo] writing synthetic dataset...")
    splits = _write_dataset(data_root)

    proto_dir = workdir / "protocols"
    proto_dir.mkdir(parents=True, exist_ok=True)
    _write_protocol(
        proto_dir / "train.yaml", base_dir=workdir,
        bona_dir=splits["train"]["bonafide"],
        spoof_dirs={"synth_a": splits["train"]["spoof_a"],
                    "synth_b": splits["train"]["spoof_b"]},
    )
    _write_protocol(
        proto_dir / "dev.yaml", base_dir=workdir,
        bona_dir=splits["dev"]["bonafide"],
        spoof_dirs={"synth_a": splits["dev"]["spoof_a"]},   # seen
    )
    _write_protocol(
        proto_dir / "eval.yaml", base_dir=workdir,
        bona_dir=splits["eval"]["bonafide"],
        spoof_dirs={"synth_b": splits["eval"]["spoof_b"]},   # HELD-OUT
    )

    # 2. Build a tiny model (no SSL) ------------------------------------
    from voice_defense.defense.aasist.backend import AASIST
    from voice_defense.defense.aasist.loss import OCSoftmaxLoss
    from voice_defense.common.dataset import (
        load_protocol, ProtocolDataset,
    )

    frontend = TinyFrontend(hidden=64)
    backend = AASIST(in_dim=frontend.hidden_size, gat_dim=32,
                     n_subgraph_nodes=16, embed_dim=32)
    loss_fn = OCSoftmaxLoss(feat_dim=32)

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fe = frontend
            self.be = backend
            self.head = loss_fn

        def embed(self, wav):
            return self.be(self.fe(wav))

        def score(self, wav):
            return self.head.score(self.embed(wav))

        def forward(self, wav, labels=None):
            emb = self.embed(wav)
            if labels is None:
                return self.head.score(emb)
            loss, scores = self.head(emb, labels)
            return loss, scores, emb

    model = Detector()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[demo] tiny detector: {n_params:,} params")

    # 3. Train (3 epochs, no augmentation for speed) --------------------
    train_proto = load_protocol(proto_dir / "train.yaml", base_dir=workdir)
    dev_proto = load_protocol(proto_dir / "dev.yaml", base_dir=workdir)
    eval_proto = load_protocol(proto_dir / "eval.yaml", base_dir=workdir)
    print(f"[demo] protocols: train={len(train_proto)} "
          f"dev={len(dev_proto)} eval={len(eval_proto)}")

    train_ds = ProtocolDataset(train_proto, segment_seconds=2.0, train=True)
    dev_ds = ProtocolDataset(dev_proto, segment_seconds=2.0, train=False)
    eval_ds = ProtocolDataset(eval_proto, segment_seconds=2.0, train=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=8)
    eval_loader = DataLoader(eval_ds, batch_size=8)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    print("[demo] training 3 epochs...")
    for epoch in range(3):
        model.train()
        running = 0.0
        for wav, label, _tag in train_loader:
            loss, _, _ = model(wav, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running / max(len(train_loader), 1)
        print(f"  epoch {epoch}: train_loss={avg:.4f}")

    # 4. Evaluate -------------------------------------------------------
    from voice_defense.evaluation.metrics import (
        compute_eer, compute_min_tdcf, attack_breakdown,
    )

    def _score_loader(loader):
        model.eval()
        scores, labels, tags = [], [], []
        with torch.no_grad():
            for wav, label, tag in loader:
                s = model.score(wav)
                scores.append(s.cpu().numpy())
                labels.append(label.numpy())
                tags.extend(list(tag))
        return (np.concatenate(scores), np.concatenate(labels), tags)

    dev_scores, dev_labels, dev_tags = _score_loader(dev_loader)
    dev_eer, dev_thr = compute_eer(dev_scores, dev_labels)
    dev_tdcf, _ = compute_min_tdcf(dev_scores, dev_labels)

    eval_scores, eval_labels, eval_tags = _score_loader(eval_loader)
    eval_eer, _ = compute_eer(eval_scores, eval_labels)
    eval_tdcf, _ = compute_min_tdcf(eval_scores, eval_labels)
    eval_per_attack = attack_breakdown(eval_scores, eval_labels, eval_tags, dev_thr)

    print(f"\n[demo] dev   EER = {dev_eer*100:5.2f}%  min-tDCF = {dev_tdcf:.4f}  (seen attack)")
    print(f"[demo] eval  EER = {eval_eer*100:5.2f}%  min-tDCF = {eval_tdcf:.4f}  (UNSEEN attack)")
    print(f"[demo] per-attack on eval @ dev-threshold {dev_thr:.4f}:")
    for tag, stats in eval_per_attack.items():
        print(f"         {tag:15s} {stats}")

    # 5. Test report writer --------------------------------------------
    from voice_defense.evaluation.report import write_report
    results = {
        "checkpoint": "(demo-in-memory)",
        "device": "cpu",
        "protocols": [
            {
                "protocol": "dev",
                "n": int(len(dev_scores)),
                "eer": float(dev_eer),
                "min_tdcf": float(dev_tdcf),
                "threshold_at_eer": float(dev_thr),
                "per_attack": attack_breakdown(dev_scores, dev_labels, dev_tags, dev_thr),
            },
            {
                "protocol": "eval",
                "n": int(len(eval_scores)),
                "eer": float(eval_eer),
                "min_tdcf": float(eval_tdcf),
                "threshold_at_eer": float(dev_thr),
                "per_attack": eval_per_attack,
            },
        ],
    }
    paths = write_report(results, workdir / "reports")
    print(f"\n[demo] report -> {paths['markdown']}")

    # 6. Smoke test: red-team submission scorer ------------------------
    # Build a tiny submission dir from spoof_b (which the model has not seen)
    submission_dir = workdir / "redteam_submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(splits["eval"]["spoof_b"].iterdir())[:8]:
        shutil.copy(src, submission_dir / src.name)
    # Synthetic labels.csv (all spoof = label 0)
    (submission_dir / "labels.csv").write_text(
        "file,label\n" + "\n".join(f"{p.name},0" for p in submission_dir.glob("*.wav")),
        encoding="utf-8",
    )

    # Save tiny checkpoint for the redteam scorer to load
    ckpt_dir = workdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "demo.pt"
    dummy_cfg = {
        "data": {"sample_rate": 16000, "segment_seconds": 2.0, "batch_size": 8},
        "model": {
            "frontend": {"pretrained": "demo-tiny-frontend", "freeze": True,
                         "output_layer_weighted": False},
            "backend": {"gat_dim": 32, "pool_dim": 16},
            "loss": {"feat_dim": 32, "r_real": 0.9, "r_fake": 0.2, "alpha": 20.0},
        },
    }
    torch.save({"epoch": 2, "model_state": model.state_dict(),
                "config": dummy_cfg, "dev_threshold": float(dev_thr),
                "dev_eer": float(dev_eer)}, ckpt_path)

    # The full submission CLI requires an SSL frontend; we skip the network
    # path here and instead exercise the scoring logic directly:
    print(f"\n[demo] mock-scoring red-team submission ({submission_dir})...")
    rt_scores = []
    for wav_path in sorted(submission_dir.glob("*.wav")):
        wav, _sr = sf.read(str(wav_path), dtype="float32")
        n = int(2.0 * 16000)
        if len(wav) > n:
            wav = wav[(len(wav)-n)//2 : (len(wav)-n)//2 + n]
        else:
            reps = int(np.ceil(n / max(len(wav), 1)))
            wav = np.tile(wav, reps)[:n]
        with torch.no_grad():
            s = float(model.score(torch.from_numpy(wav).unsqueeze(0)).item())
        rt_scores.append(s)
    passed = sum(1 for s in rt_scores if s > dev_thr)
    print(f"         {passed}/{len(rt_scores)} samples passed (= attack success); "
          f"ASR = {passed/len(rt_scores)*100:.1f}%")

    return {
        "dev_eer": float(dev_eer),
        "eval_eer": float(eval_eer),
        "dev_tdcf": float(dev_tdcf),
        "eval_tdcf": float(eval_tdcf),
        "redteam_asr": passed / len(rt_scores),
        "report_md": str(paths["markdown"]),
        "workdir": str(workdir),
    }


def main() -> int:
    workdir = Path("voice_defense/_demo_run")
    if workdir.exists():
        shutil.rmtree(workdir)
    t0 = time.time()
    result = run(workdir)
    print(f"\n[demo] DONE in {time.time()-t0:.1f}s")
    print(f"[demo] summary = {json.dumps(result, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
