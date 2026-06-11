"""Tandem ASV + CM evaluation.

Problem: whether a clone passes the speaker verifier (ASV) and whether the
countermeasure (CM / spoof detector) catches it were measured separately. The
real threat is both happening at once.

Solution: run each spoof (cloned) trial through ASV and CM together and tally
the four outcomes:

                       CM evades (undetected)     CM catches (detected)
    ASV accept         ★ full bypass (worst)      ASV broken, CM defends
    ASV reject           CM-only would fail        both defended

Key metrics:
    asr_asv     = P(ASV accepts)                          ASV-only bypass rate
    asr_tandem  = P(ASV accepts AND CM evades)  ★         combined bypass rate

Story: a high ``asr_asv`` with a low ``asr_tandem`` is the quantitative proof
that adding the CM contributed to defense.

Both scorers are injected, so the module is testable without models:
  asv_score_fn(enroll_path, test_path) -> similarity (>= threshold == accept)
  cm_score_fn(test_path)               -> spoofness  (>= threshold == detected)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import json
from pathlib import Path

from ..common.trial_protocol import Trial, KIND_SPOOF

AsvScoreFn = Callable[[str, str], float]
CmScoreFn = Callable[[str], float]

# four-quadrant keys
Q_FULL_BYPASS = "full_bypass"          # ASV accept + CM evade  (worst case)
Q_CM_DEFENDS = "cm_defends"            # ASV accept + CM catch
Q_CM_ONLY = "cm_only_would_fail"       # ASV reject + CM evade
Q_BOTH_DEFEND = "both_defended"        # ASV reject + CM catch


@dataclass
class TandemResult:
    n: int
    asv_threshold: float
    cm_threshold: float
    counts: dict = field(default_factory=dict)
    asr_asv: float = 0.0
    asr_cm_evade: float = 0.0
    asr_tandem: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "asv_threshold": self.asv_threshold,
            "cm_threshold": self.cm_threshold,
            "asr_asv": self.asr_asv,
            "asr_cm_evade": self.asr_cm_evade,
            "asr_tandem": self.asr_tandem,
            "quadrants": self.counts,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def evaluate_tandem(
    spoof_trials: Sequence[Trial],
    asv_score_fn: AsvScoreFn,
    asv_threshold: float,
    cm_score_fn: CmScoreFn,
    cm_threshold: float,
) -> TandemResult:
    """Evaluate cloned (spoof) trials against ASV and CM jointly.

    Args:
        spoof_trials: trials with kind == "spoof" (others are ignored).
        asv_score_fn: similarity; ``>= asv_threshold`` means ASV accepts.
        asv_threshold: calibrated operating point (see ``calibrate_asv``).
        cm_score_fn: spoofness; ``>= cm_threshold`` means CM detects the spoof.
        cm_threshold: CM operating point.
    """
    trials = [t for t in spoof_trials if t.kind == KIND_SPOOF]
    counts = {Q_FULL_BYPASS: 0, Q_CM_DEFENDS: 0, Q_CM_ONLY: 0, Q_BOTH_DEFEND: 0}

    n_asv_accept = 0
    n_cm_evade = 0
    n_tandem = 0

    for t in trials:
        asv_accept = float(asv_score_fn(t.enroll_path, t.test_path)) >= asv_threshold
        cm_detect = float(cm_score_fn(t.test_path)) >= cm_threshold
        cm_evade = not cm_detect

        n_asv_accept += asv_accept
        n_cm_evade += cm_evade
        if asv_accept and cm_evade:
            counts[Q_FULL_BYPASS] += 1
            n_tandem += 1
        elif asv_accept and not cm_evade:
            counts[Q_CM_DEFENDS] += 1
        elif (not asv_accept) and cm_evade:
            counts[Q_CM_ONLY] += 1
        else:
            counts[Q_BOTH_DEFEND] += 1

    n = len(trials)
    denom = n if n else 1
    return TandemResult(
        n=n, asv_threshold=asv_threshold, cm_threshold=cm_threshold, counts=counts,
        asr_asv=n_asv_accept / denom,
        asr_cm_evade=n_cm_evade / denom,
        asr_tandem=n_tandem / denom,
    )


def cm_score_fn_from_detector(model, model_type, *, device="cpu", **kw) -> CmScoreFn:
    """Adapt ``defense.alt.detector.detect_single`` into a CmScoreFn.

    Returns spoofness = ``spoof_score`` so that ``>= threshold`` == detected.
    Imported lazily to avoid a hard torch dependency at module import time.
    """
    from ..defense.alt.detector import detect_single

    def _fn(test_path: str) -> float:
        return float(detect_single(test_path, model, model_type, device=device, **kw)["spoof_score"])

    return _fn
