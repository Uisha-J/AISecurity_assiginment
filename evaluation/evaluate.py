"""Per-protocol evaluation. Loads a checkpoint, scores every file in each
protocol, reports EER + min t-DCF + per-attack breakdown.

The cardinal sin in anti-spoofing is reporting one number averaged across
protocols. We always report each protocol separately, and we always show
the per-attack breakdown — that's where you see WHICH attack fooled you.
"""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data_pipeline import ProtocolDataset, load_protocol
from ..defense.model import SpoofDetector, SpoofDetectorConfig
from .metrics import compute_eer, compute_min_tdcf, attack_breakdown


def _load_model(ckpt_path: str, device: str) -> tuple[SpoofDetector, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    m = cfg["model"]
    detector_cfg = SpoofDetectorConfig(
        ssl_pretrained=m["frontend"]["pretrained"],
        ssl_freeze=m["frontend"].get("freeze", True),
        ssl_weighted=m["frontend"].get("output_layer_weighted", True),
        gat_dim=m["backend"].get("gat_dim", 64),
        n_subgraph_nodes=m["backend"].get("pool_dim", 32),
        embed_dim=m["loss"].get("feat_dim", 128),
        sample_rate=cfg["data"]["sample_rate"],
        r_real=m["loss"].get("r_real", 0.9),
        r_fake=m["loss"].get("r_fake", 0.2),
        alpha=m["loss"].get("alpha", 20.0),
    )
    model = SpoofDetector(detector_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg


def _score_protocol(model: SpoofDetector, protocol_path: str, cfg: dict,
                    device: str) -> dict:
    proto = load_protocol(protocol_path)
    ds = ProtocolDataset(
        proto,
        sample_rate=cfg["data"]["sample_rate"],
        segment_seconds=cfg["data"]["segment_seconds"],
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 2),
    )
    scores, labels, tags = [], [], []
    with torch.no_grad():
        for wav, label, tag in loader:
            wav = wav.to(device)
            s = model.score(wav)
            scores.append(s.cpu().numpy())
            labels.append(label.numpy())
            tags.extend(list(tag))
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    eer, thr = compute_eer(scores, labels)
    min_tdcf, _ = compute_min_tdcf(scores, labels)
    return {
        "protocol": proto.name,
        "n": int(len(scores)),
        "eer": float(eer),
        "min_tdcf": float(min_tdcf),
        "threshold_at_eer": float(thr),
        "per_attack": attack_breakdown(scores, labels, tags, thr),
        "_scores": scores.tolist(),
        "_labels": labels.tolist(),
        "_tags": tags,
    }


def evaluate_checkpoint(
    ckpt_path: str,
    protocol_paths: list[str],
    device: Optional[str] = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_model(ckpt_path, device)
    results = {"checkpoint": ckpt_path, "device": device, "protocols": []}
    for p in protocol_paths:
        print(f"[eval] scoring protocol: {p}")
        r = _score_protocol(model, p, cfg, device)
        # Drop raw arrays from the headline result; saved separately on disk.
        headline = {k: v for k, v in r.items() if not k.startswith("_")}
        print(f"  {headline['protocol']}: "
              f"EER={headline['eer']*100:.2f}% "
              f"min-tDCF={headline['min_tdcf']:.4f} "
              f"N={headline['n']}")
        for tag, stats in headline["per_attack"].items():
            print(f"    {tag:20s} {stats}")
        results["protocols"].append(r)
    return results
