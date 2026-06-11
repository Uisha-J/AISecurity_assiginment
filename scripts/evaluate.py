"""CLI: evaluate a checkpoint against one or more protocols.

Example:
    python -m voice_defense.scripts.evaluate \
        --checkpoint checkpoints/best.pt \
        --protocols voice_defense/protocols/seen.yaml \
                    voice_defense/protocols/unseen.yaml \
                    voice_defense/protocols/wild.yaml \
        --report-dir reports
"""

from __future__ import annotations
import argparse

from ..evaluation.eval_aasist import evaluate_checkpoint
from ..evaluation.report import write_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--protocols", nargs="+", required=True)
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    results = evaluate_checkpoint(
        ckpt_path=args.checkpoint,
        protocol_paths=args.protocols,
        device=args.device,
    )
    paths = write_report(results, args.report_dir)
    print(f"[report] markdown -> {paths['markdown']}")
    print(f"[report] json     -> {paths['json']}")


if __name__ == "__main__":
    main()
