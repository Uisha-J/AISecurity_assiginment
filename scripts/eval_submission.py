"""CLI: score a red-team submission directory.

Example:
    python -m voice_defense.scripts.eval_submission \
        --submission-dir redteam/submissions/team_a \
        --checkpoint checkpoints/best.pt
"""

from __future__ import annotations
import argparse

from ..redteam.evaluate_submission import evaluate_submission


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--submission-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="reports/redteam")
    p.add_argument("--threshold", type=float, default=None,
                   help="If omitted, uses dev_threshold saved in the checkpoint.")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    evaluate_submission(
        submission_dir=args.submission_dir,
        ckpt_path=args.checkpoint,
        out_dir=args.out_dir,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
