"""Inference and evaluation for LCNN / RawNet2 detectors."""

import logging
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from .models import LCNN, RawNet2
from ...common.features import extract_lfcc
from ...common.audio import load_audio
from ...evaluation.metrics import compute_eer

logger = logging.getLogger(__name__)


def load_alt_detector(checkpoint_path, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = LCNN() if ckpt["model_type"] == "lcnn" else RawNet2()
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt["model_type"]


def detect_single(audio_path, model, model_type, device="cuda", max_len=64000, n_lfcc=60):
    device = device if torch.cuda.is_available() else "cpu"
    wav, _ = load_audio(audio_path, target_sr=16000)
    if model_type == "lcnn":
        wav = wav[:max_len]
        feat = torch.from_numpy(extract_lfcc(wav, n_lfcc=n_lfcc)).float().unsqueeze(0).to(device)
    else:
        wav = wav[:max_len] if len(wav) > max_len else np.pad(wav, (0, max_len - len(wav)))
        feat = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(feat), dim=1).squeeze()
    return {"prediction": p.argmax().item(), "bonafide_score": p[0].item(), "spoof_score": p[1].item()}


def evaluate_on_clones(checkpoint_path, cloned_paths, real_paths, output_root, device="cuda"):
    model, mtype = load_alt_detector(checkpoint_path, device)
    results = []
    for p in tqdm(cloned_paths, desc="Detecting clones"):
        r = detect_single(p, model, mtype, device)
        r.update(audio_path=p, true_label=1)
        results.append(r)
    for p in tqdm(real_paths, desc="Detecting real"):
        r = detect_single(p, model, mtype, device)
        r.update(audio_path=p, true_label=0)
        results.append(r)
    df = pd.DataFrame(results)
    Path(output_root, "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(output_root) / "results" / "alt_detector_eval.csv", index=False)
    return {
        "accuracy": (df["prediction"].values == df["true_label"].values).mean(),
        "eer": compute_eer(df["true_label"].values, df["spoof_score"].values),
        "clone_detection_rate": (df[df["true_label"] == 1]["prediction"] == 1).mean(),
        "false_rejection_rate": (df[df["true_label"] == 0]["prediction"] == 1).mean(),
    }
