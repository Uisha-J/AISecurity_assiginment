"""Score every wav in a red-team submission directory and write a report.

Submission layout (any of these is accepted):

  submissions/team_x/
    foo.wav
    bar.wav
    ...
    [optional] labels.csv      with columns: file,label  (1=bonafide,0=spoof)

We don't require labels. If absent, we report the score distribution and
the count of files that would pass the calibrated threshold.
"""

from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import soundfile as sf

from ..defense.aasist.model import SpoofDetector, SpoofDetectorConfig
from ..evaluation.metrics import compute_eer, compute_min_tdcf


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
    return wav.astype(np.float32)


def _build_model(ckpt_path: str, device: str) -> tuple[SpoofDetector, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
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
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg


def _maybe_load_labels(submission_dir: Path) -> Optional[dict[str, int]]:
    labels_csv = submission_dir / "labels.csv"
    if not labels_csv.exists():
        return None
    out: dict[str, int] = {}
    with labels_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["file"].strip()] = int(row["label"])
    return out


def evaluate_submission(
    submission_dir: str | Path,
    ckpt_path: str,
    out_dir: str | Path = "reports/redteam",
    threshold: Optional[float] = None,
    device: Optional[str] = None,
) -> dict:
    submission_dir = Path(submission_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _build_model(ckpt_path, device)

    if threshold is None:
        # Use threshold saved with checkpoint, else fall back to 0.5
        ckpt = torch.load(ckpt_path, map_location="cpu")
        threshold = float(ckpt.get("dev_threshold", 0.5))
        print(f"[redteam] using dev-set threshold = {threshold:.4f}")

    wav_files = sorted(submission_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files in {submission_dir}")

    labels_map = _maybe_load_labels(submission_dir)
    print(f"[redteam] {len(wav_files)} files; labels = {'yes' if labels_map else 'no'}")

    sr = cfg["data"]["sample_rate"]
    seg = cfg["data"]["segment_seconds"]
    batch_size = cfg["data"].get("batch_size", 16)

    scores: list[float] = []
    files: list[str] = []
    labels: list[int] = []

    batch_wavs: list[np.ndarray] = []
    batch_files: list[str] = []

    def _flush():
        if not batch_wavs:
            return
        x = torch.from_numpy(np.stack(batch_wavs)).to(device)
        with torch.no_grad():
            s = model.score(x).cpu().numpy()
        scores.extend(s.tolist())
        files.extend(batch_files)
        batch_wavs.clear()
        batch_files.clear()

    for p in wav_files:
        try:
            wav = _load_wav(p, sr, seg)
        except Exception as e:
            print(f"[redteam] skip {p.name}: {e}")
            continue
        batch_wavs.append(wav)
        batch_files.append(p.name)
        if labels_map is not None:
            labels.append(labels_map.get(p.name, -1))
        if len(batch_wavs) >= batch_size:
            _flush()
    _flush()

    scores_arr = np.array(scores)
    report: dict = {
        "submission": str(submission_dir),
        "checkpoint": str(ckpt_path),
        "threshold": threshold,
        "n_files": len(files),
        "score_stats": {
            "mean": float(scores_arr.mean()),
            "std": float(scores_arr.std()),
            "min": float(scores_arr.min()),
            "max": float(scores_arr.max()),
        },
        "n_above_threshold": int((scores_arr > threshold).sum()),
        "files": [{"file": f, "score": s} for f, s in zip(files, scores)],
    }
    if labels_map is not None and len(labels) == len(scores):
        labels_arr = np.array(labels)
        mask = labels_arr != -1
        if mask.sum() > 0 and len(set(labels_arr[mask].tolist())) > 1:
            eer, eer_thr = compute_eer(scores_arr[mask], labels_arr[mask])
            tdcf, _ = compute_min_tdcf(scores_arr[mask], labels_arr[mask])
            report["with_labels"] = {
                "n_labeled": int(mask.sum()),
                "eer": float(eer),
                "min_tdcf": float(tdcf),
                "threshold_at_eer": float(eer_thr),
            }
            # Attack success rate on the SPOOF subset
            spoof_mask = mask & (labels_arr == 0)
            if spoof_mask.any():
                attack_success = float(
                    (scores_arr[spoof_mask] > threshold).mean()
                )
                report["with_labels"]["attack_success_rate"] = attack_success

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"redteam-{submission_dir.name}-{ts}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[redteam] wrote {json_path}")
    print(f"[redteam] n_above_threshold = {report['n_above_threshold']} / {report['n_files']}")
    if "with_labels" in report:
        print(f"[redteam] EER = {report['with_labels']['eer']*100:.2f}%  "
              f"min-tDCF = {report['with_labels']['min_tdcf']:.4f}")
    return report
