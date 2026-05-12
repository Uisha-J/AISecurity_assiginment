"""Red-team submission interface.

The attacker drops `.wav` files into a submission directory (optionally with
labels.csv to compute their attack success rate). We score every file with
the trained model and write a report.

Even without labels, we report score histograms — letting the red team see
which of their samples crossed the threshold (= "got through").
"""

from .evaluate_submission import evaluate_submission

__all__ = ["evaluate_submission"]
