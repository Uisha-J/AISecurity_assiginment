"""CLI: score a single wav file (quick smoke test of a checkpoint).

Example:
    python -m voice_defense.scripts.infer \
        --checkpoint checkpoints/best.pt \
        --wav suspect.wav
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from ..defense.model import SpoofDetector, SpoofDetectorConfig


def _load(ckpt: str, device: str):
    state = torch.load(ckpt, map_location=device)
    cfg = state["config"]
    m = cfg["model"]
    dcfg = SpoofDetectorConfig(
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
    model = SpoofDetector(dcfg)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    return model, cfg, float(state.get("dev_threshold", 0.5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, threshold = _load(args.checkpoint, device)
    sr = cfg["data"]["sample_rate"]
    seg = cfg["data"]["segment_seconds"]

    wav, in_sr = sf.read(args.wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if in_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=in_sr, target_sr=sr)
    n = int(sr * seg)
    if len(wav) >= n:
        start = (len(wav) - n) // 2
        wav = wav[start : start + n]
    else:
        reps = int(np.ceil(n / max(len(wav), 1)))
        wav = np.tile(wav, reps)[:n]

    x = torch.from_numpy(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        score = float(model.score(x).item())
    verdict = "BONAFIDE" if score > threshold else "SPOOF"
    print(f"file = {args.wav}")
    print(f"score (higher = bonafide) = {score:.4f}")
    print(f"threshold = {threshold:.4f}")
    print(f"verdict = {verdict}")


if __name__ == "__main__":
    main()
