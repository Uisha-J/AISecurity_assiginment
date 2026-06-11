"""CLI: train the spoof detector.

Example:
    python -m voice_defense.scripts.train --config voice_defense/configs/default.yaml
"""

from __future__ import annotations
import argparse

from ..defense.aasist.train import TrainArgs, train_one_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    ta = TrainArgs.from_yaml(args.config)
    result = train_one_run(ta)
    print(f"[done] best EER = {result['best_eer']*100:.2f}%")
    print(f"[done] checkpoints at {result['ckpt_dir']}")


if __name__ == "__main__":
    main()
