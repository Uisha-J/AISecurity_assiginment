"""Verify cloned voices against speaker verification and compute ASR."""

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .speaker_verifier import SpeakerVerifier

logger = logging.getLogger(__name__)


def verify_clones_experiment(
    cloned_files, reference_samples, output_root,
    threshold_sweep=None, verification_model="speechbrain/spkrec-ecapa-voxceleb",
    device="cuda",
):
    if threshold_sweep is None:
        threshold_sweep = [0.15, 0.20, 0.25, 0.30, 0.35]

    results_dir = Path(output_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    verifier = SpeakerVerifier(model_source=verification_model, device=device)

    results = []
    for item in tqdm(cloned_files, desc="Verifying clones"):
        spk_id = item["speaker_id"]
        enrollment_path = reference_samples[spk_id][max(reference_samples[spk_id])]
        score = verifier.similarity(enrollment_path, item["cloned_path"])
        for threshold in threshold_sweep:
            results.append({
                "speaker_id": spk_id, "duration": item["duration"], "text_id": item["text_id"],
                "threshold": threshold, "similarity_score": score, "verified": score >= threshold,
                "cloned_path": item["cloned_path"], "enrollment_path": enrollment_path,
            })

    df = pd.DataFrame(results)
    df.to_csv(results_dir / "attack_scores.csv", index=False)
    for t in threshold_sweep:
        logger.info(f"Threshold {t:.2f}: ASR = {df[df['threshold']==t]['verified'].mean():.4f}")
    return df
