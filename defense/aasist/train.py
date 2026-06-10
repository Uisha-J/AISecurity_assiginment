"""Training loop.

Usage (programmatic):
    args = TrainArgs.from_yaml("configs/default.yaml")
    train_one_run(args)

The CLI entry point lives in scripts/train.py.
"""

from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ...common import (
    ProtocolDataset,
    load_protocol,
    build_augmentation_chain,
)
from ...common.augment import AugmentChainSpec
from .model import SpoofDetector, SpoofDetectorConfig
from ...evaluation.metrics import compute_eer


@dataclass
class TrainArgs:
    config_path: str
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainArgs":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(config_path=path, raw=raw)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_model(cfg: dict) -> SpoofDetector:
    m = cfg["model"]
    detector_cfg = SpoofDetectorConfig(
        ssl_pretrained=m["frontend"]["pretrained"],
        ssl_freeze=m["frontend"].get("freeze", True),
        ssl_weighted=m["frontend"].get("output_layer_weighted", True),
        gat_dim=m["backend"].get("gat_dim", 64),
        n_subgraph_nodes=m["backend"].get("pool_dim", 32),
        embed_dim=m["loss"].get("feat_dim", 128),
        dropout=0.1,
        sample_rate=cfg["data"]["sample_rate"],
        r_real=m["loss"].get("r_real", 0.9),
        r_fake=m["loss"].get("r_fake", 0.2),
        alpha=m["loss"].get("alpha", 20.0),
    )
    return SpoofDetector(detector_cfg)


def _build_augment(cfg: dict, rng: random.Random):
    a = cfg.get("augmentation", {})
    spec = AugmentChainSpec(
        codec_enabled=a.get("codec", {}).get("enabled", True),
        codec_prob=a.get("codec", {}).get("prob", 0.5),
        codec_codecs=tuple(a.get("codec", {}).get("codecs",
            ["opus", "mp3", "g711_alaw", "g711_ulaw"])),
        rir_enabled=a.get("rir", {}).get("enabled", True),
        rir_prob=a.get("rir", {}).get("prob", 0.3),
        rir_dir=a.get("rir", {}).get("rir_dir", "data/augment/rirs"),
        noise_enabled=a.get("noise", {}).get("enabled", True),
        noise_prob=a.get("noise", {}).get("prob", 0.4),
        noise_dir=a.get("noise", {}).get("noise_dir", "data/augment/musan"),
        noise_snr_db=tuple(a.get("noise", {}).get("snr_db", [5, 20])),
        rawboost_enabled=a.get("rawboost", {}).get("enabled", True),
        rawboost_prob=a.get("rawboost", {}).get("prob", 0.5),
        rawboost_algos=tuple(a.get("rawboost", {}).get("algo", [1, 2, 3])),
    )
    return build_augmentation_chain(spec, rng=rng)


def _evaluate_dev(model: SpoofDetector, loader: DataLoader, device: str) -> dict:
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for wav, label, _tag in loader:
            wav = wav.to(device)
            s = model.score(wav)
            scores.append(s.cpu().numpy())
            labels.append(label.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    eer, thr = compute_eer(scores, labels)
    return {"eer": eer, "threshold": thr, "n": len(scores)}


def train_one_run(args: TrainArgs) -> dict:
    cfg = args.raw
    _set_seed(cfg.get("seed", 1337))
    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU.")
        device = "cpu"

    rng = random.Random(cfg.get("seed", 1337))

    # ---- data
    train_proto = load_protocol(cfg["data"]["train_protocol"])
    dev_proto = load_protocol(cfg["data"]["dev_protocol"])
    print(f"[data] train protocol '{train_proto.name}': "
          f"{len(train_proto)} files, classes={train_proto.class_counts()}")
    print(f"[data] dev   protocol '{dev_proto.name}': "
          f"{len(dev_proto)} files, classes={dev_proto.class_counts()}")

    augment = _build_augment(cfg, rng=rng)
    train_ds = ProtocolDataset(
        train_proto,
        sample_rate=cfg["data"]["sample_rate"],
        segment_seconds=cfg["data"]["segment_seconds"],
        train=True,
        augment=augment,
    )
    dev_ds = ProtocolDataset(
        dev_proto,
        sample_rate=cfg["data"]["sample_rate"],
        segment_seconds=cfg["data"]["segment_seconds"],
        train=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        drop_last=True,
        pin_memory=device == "cuda",
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=device == "cuda",
    )

    # ---- model
    model = _build_model(cfg).to(device)
    print(f"[model] params = {sum(p.numel() for p in model.parameters()):,} "
          f"(trainable = {sum(p.numel() for p in model.parameters() if p.requires_grad):,})")

    # ---- optim
    t = cfg["training"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t["epochs"], eta_min=t["lr"] * 0.01
    )

    ckpt_dir = Path(t.get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_every = t.get("log_every", 50)
    grad_clip = t.get("grad_clip", 5.0)
    patience = t.get("early_stop_patience", 5)

    best_eer = float("inf")
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(t["epochs"]):
        model.train()
        running = 0.0
        t0 = time.time()
        for step, (wav, label, _tag) in enumerate(train_loader):
            wav = wav.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            loss, scores, emb = model(wav, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            running += loss.item()

            if (step + 1) % log_every == 0:
                print(f"  ep{epoch} step{step+1}/{len(train_loader)} "
                      f"loss={running / (step+1):.4f}")

        scheduler.step()
        dev_metrics = _evaluate_dev(model, dev_loader, device)
        epoch_time = time.time() - t0
        print(f"[epoch {epoch}] train_loss={running/len(train_loader):.4f} "
              f"dev_eer={dev_metrics['eer']*100:.2f}% time={epoch_time:.1f}s")
        history.append({
            "epoch": epoch,
            "train_loss": running / len(train_loader),
            "dev_eer": dev_metrics["eer"],
            "dev_threshold": dev_metrics["threshold"],
        })

        improved = dev_metrics["eer"] < best_eer
        if improved:
            best_eer = dev_metrics["eer"]
            bad_epochs = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": cfg,
                "dev_eer": best_eer,
                "dev_threshold": dev_metrics["threshold"],
            }
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  -> new best, saved to {ckpt_dir/'best.pt'}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[early-stop] {patience} epochs without improvement.")
                break

    torch.save(
        {"epoch": epoch, "model_state": model.state_dict(), "config": cfg},
        ckpt_dir / "last.pt",
    )
    return {"best_eer": best_eer, "history": history, "ckpt_dir": str(ckpt_dir)}
