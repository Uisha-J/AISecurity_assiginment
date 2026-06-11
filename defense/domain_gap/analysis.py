"""Domain gap analysis: FID on mel-spectrogram distributions + t-SNE."""

import json
import logging
from pathlib import Path

import numpy as np
from scipy import linalg
from sklearn.manifold import TSNE
from tqdm import tqdm

from ...common.features import extract_mel_spectrogram
from ...common.audio import load_audio

logger = logging.getLogger(__name__)


def _spectrogram_stats(audio_paths, sr=16000, n_mels=80, max_files=500):
    feats = []
    for p in tqdm(audio_paths[:max_files], desc="Extracting spectrograms"):
        try:
            wav, _ = load_audio(p, target_sr=sr)
            feats.append(extract_mel_spectrogram(wav, sr=sr, n_mels=n_mels).mean(axis=1))
        except Exception as e:
            logger.warning(f"Failed: {p}: {e}")
    arr = np.array(feats)
    return arr.mean(0), np.cov(arr, rowvar=False), arr


def compute_fid(mu1, sig1, mu2, sig2):
    diff = mu1 - mu2
    cm, _ = linalg.sqrtm(sig1 @ sig2, disp=False)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(diff @ diff + np.trace(sig1 + sig2 - 2 * cm))


def compute_domain_gap(asvspoof_paths, cloned_paths, real_paths, output_root, max_files=500):
    results_dir = Path(output_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    mu_a, s_a, f_a = _spectrogram_stats(asvspoof_paths, max_files=max_files)
    mu_c, s_c, f_c = _spectrogram_stats(cloned_paths, max_files=max_files)
    mu_r, s_r, f_r = _spectrogram_stats(real_paths, max_files=max_files)

    gap = {
        "fid_asvspoof_vs_cloned": compute_fid(mu_a, s_a, mu_c, s_c),
        "fid_asvspoof_vs_real": compute_fid(mu_a, s_a, mu_r, s_r),
        "fid_real_vs_cloned": compute_fid(mu_r, s_r, mu_c, s_c),
    }

    n_a, n_c, n_r = min(len(f_a), 200), min(len(f_c), 200), min(len(f_r), 200)
    combined = np.vstack([f_a[:n_a], f_c[:n_c], f_r[:n_r]])
    labels = ["ASVspoof"] * n_a + ["XTTS Clone"] * n_c + ["Real"] * n_r
    tsne_coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(combined)

    with open(results_dir / "domain_gap_fid.json", "w") as f:
        json.dump(gap, f, indent=2)
    np.savez(results_dir / "tsne_data.npz", coords=tsne_coords, labels=np.array(labels))
    return {**gap, "tsne_coords": tsne_coords, "tsne_labels": labels}
