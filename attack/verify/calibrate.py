"""ASV operating-point calibration.

Problem: measuring attack success at an arbitrary similarity threshold
(e.g. the 0.15–0.35 sweep in ``asr_calculator``) is meaningless — the number
has no relation to how the verifier would actually be deployed.

Solution: fix the ASV operating point at its Equal Error Rate (EER) using
genuine vs zero-effort-impostor trials. The cloned-voice ASR is then reported
*at that calibrated threshold*, so "the clone passed a real operating
authenticator" becomes a defensible claim.

The scoring function is injected, so this module needs no model to be tested:
pass ``SpeakerVerifier.similarity`` in production, or a mock in unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import json
from pathlib import Path

import numpy as np

from ...evaluation.metrics import compute_eer
from ...common.trial_protocol import TrialSet, KIND_GENUINE, KIND_IMPOSTOR

# score_fn(enroll_path, test_path) -> similarity (higher == more same-speaker)
ScoreFn = Callable[[str, str], float]


@dataclass
class CalibrationResult:
    eer: float
    threshold: float                       # threshold at EER operating point
    n_genuine: int
    n_impostor: int
    genuine_scores: list[float] = field(default_factory=list)
    impostor_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "eer": self.eer,
            "threshold_at_eer": self.threshold,
            "n_genuine": self.n_genuine,
            "n_impostor": self.n_impostor,
            "genuine_mean": float(np.mean(self.genuine_scores)) if self.genuine_scores else None,
            "impostor_mean": float(np.mean(self.impostor_scores)) if self.impostor_scores else None,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def calibrate_asv(trial_set: TrialSet, score_fn: ScoreFn) -> CalibrationResult:
    """Calibrate the ASV threshold to its EER using genuine vs impostor trials.

    Spoof trials are intentionally excluded — calibration must use only
    zero-effort impostors so the operating point reflects normal deployment.
    """
    scores: list[float] = []
    labels: list[int] = []
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []

    for t in trial_set.trials:
        if t.kind == KIND_GENUINE:
            s = float(score_fn(t.enroll_path, t.test_path))
            scores.append(s); labels.append(1); genuine_scores.append(s)
        elif t.kind == KIND_IMPOSTOR:
            s = float(score_fn(t.enroll_path, t.test_path))
            scores.append(s); labels.append(0); impostor_scores.append(s)

    if not genuine_scores or not impostor_scores:
        raise ValueError(
            "Calibration needs both genuine and impostor trials "
            f"(got {len(genuine_scores)} genuine, {len(impostor_scores)} impostor)."
        )

    eer, threshold = compute_eer(np.asarray(scores), np.asarray(labels))
    return CalibrationResult(
        eer=eer, threshold=threshold,
        n_genuine=len(genuine_scores), n_impostor=len(impostor_scores),
        genuine_scores=genuine_scores, impostor_scores=impostor_scores,
    )


def calibrate_with_verifier(trial_set: TrialSet, verifier) -> CalibrationResult:
    """Convenience wrapper using a ``SpeakerVerifier`` instance."""
    return calibrate_asv(trial_set, verifier.similarity)
