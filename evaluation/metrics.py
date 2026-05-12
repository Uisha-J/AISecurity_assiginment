"""Spoof detection metrics: EER and min t-DCF.

EER (Equal Error Rate): operating point where FAR == FRR.
t-DCF: cost function from ASVspoof challenges that combines CM and ASV errors.
       We implement the simplified min-tDCF variant (Kinnunen et al., 2020),
       which only requires CM scores + fixed ASV operating-point assumptions.

Higher score == more likely bonafide.
"""

from __future__ import annotations
from typing import Optional

import numpy as np


def compute_det(scores: np.ndarray, labels: np.ndarray):
    """Return (frrs, fars, thresholds) for a DET curve.
    labels: 1 == bonafide (target), 0 == spoof (nontarget).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    order = np.argsort(scores)
    scores_s = scores[order]
    labels_s = labels[order]

    n_target = (labels_s == 1).sum()
    n_nontarget = (labels_s == 0).sum()
    if n_target == 0 or n_nontarget == 0:
        raise ValueError("DET needs both target and non-target trials.")

    # Cumulative counts of target/nontarget at or below each threshold
    cum_target = np.cumsum(labels_s == 1)
    cum_nontarget = np.cumsum(labels_s == 0)

    # FRR: fraction of TARGET trials with score <= threshold (rejected)
    frrs = cum_target / n_target
    # FAR: fraction of NONTARGET trials with score >  threshold (accepted)
    fars = 1.0 - cum_nontarget / n_nontarget
    thresholds = scores_s
    return frrs, fars, thresholds


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and the threshold at which it occurs."""
    frrs, fars, thresholds = compute_det(scores, labels)
    abs_diff = np.abs(frrs - fars)
    idx = int(np.argmin(abs_diff))
    eer = float((frrs[idx] + fars[idx]) / 2.0)
    return eer, float(thresholds[idx])


def compute_min_tdcf(
    cm_scores: np.ndarray,
    labels: np.ndarray,
    *,
    p_target: float = 0.05,        # ASVspoof default prior
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    c_fa_asv: float = 10.0,
    c_miss_asv: float = 1.0,
    p_miss_asv: float = 0.01,      # assumed ASV miss rate
    p_fa_asv: float = 0.01,        # assumed ASV false-accept on bonafide nontarget
) -> tuple[float, float]:
    """Simplified min t-DCF (CM-only assumption, ASV operating point fixed).

    Following Kinnunen et al. (2020), Eq. (3-4) approximation:
        tDCF(theta) = C0 + C1 * P_miss^CM(theta) + C2 * P_fa^CM(theta)
    where C1, C2 are derived from ASV behaviour and costs.

    Returns (min_tdcf, threshold_at_min).
    """
    frrs, fars, thresholds = compute_det(cm_scores, labels)
    # CM error rates
    p_miss = frrs                       # bonafide rejected
    p_fa = fars                         # spoof accepted

    C1 = c_miss * p_target * (1 - p_miss_asv)
    C2 = c_fa * (1 - p_target) * p_fa_asv
    C0 = c_miss_asv * p_target * p_miss_asv + c_fa_asv * (1 - p_target) * 0.0

    tdcf = C0 + C1 * p_miss + C2 * p_fa
    # Normalize so the trivial systems get tdcf = 1
    tdcf_default = min(C1, C2) + C0
    if tdcf_default > 0:
        tdcf = tdcf / tdcf_default

    idx = int(np.argmin(tdcf))
    return float(tdcf[idx]), float(thresholds[idx])


def attack_breakdown(
    scores: np.ndarray,
    labels: np.ndarray,
    tags: list[str],
    threshold: float,
) -> dict:
    """Per-attack-tag FAR at the global threshold."""
    out: dict[str, dict] = {}
    tags_arr = np.asarray(tags)
    # bonafide
    mask_b = labels == 1
    if mask_b.any():
        out["bonafide"] = {
            "n": int(mask_b.sum()),
            "frr": float((scores[mask_b] <= threshold).mean()),
        }
    # per-spoof
    for tag in sorted(set(tags_arr[labels == 0])):
        m = (labels == 0) & (tags_arr == tag)
        if m.any():
            out[tag] = {
                "n": int(m.sum()),
                "far": float((scores[m] > threshold).mean()),
            }
    return out
