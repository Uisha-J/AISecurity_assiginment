"""CLI: build a controlled red-team attack submission.

Example:
    python -m voice_defense.scripts.redteam_attack \
        --input-dir data/spoof_self/train \
        --out-dir redteam/submissions/team_a \
        --checkpoint checkpoints/best.pt \
        --variants-per-file 12
"""

from __future__ import annotations

import argparse

from ..attack.redteam_system import RedTeamAttackConfig, generate_redteam_submission


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True,
                   help="Directory containing authorized spoof .wav files to mutate.")
    p.add_argument("--out-dir", required=True,
                   help="Submission directory to write .wav files and labels.csv.")
    p.add_argument("--checkpoint", default=None,
                   help="Optional defender checkpoint. If set, keep highest-scoring variants.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Optional pass threshold. Defaults to checkpoint dev_threshold.")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--segment-seconds", type=float, default=4.0)
    p.add_argument("--variants-per-file", type=int, default=8)
    p.add_argument("--keep-all", action="store_true",
                   help="Write all generated variants instead of only the best per source file.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None)
    p.add_argument("--whitebox-steps", type=int, default=0,
                   help="Optional PGD steps against the checkpoint score. Default is disabled.")
    p.add_argument("--whitebox-epsilon", type=float, default=0.002)
    p.add_argument("--whitebox-step-size", type=float, default=0.0005)
    args = p.parse_args()

    manifest = generate_redteam_submission(RedTeamAttackConfig(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        variants_per_file=args.variants_per_file,
        keep_all=args.keep_all,
        seed=args.seed,
        device=args.device,
        whitebox_steps=args.whitebox_steps,
        whitebox_epsilon=args.whitebox_epsilon,
        whitebox_step_size=args.whitebox_step_size,
    ))
    print(f"[redteam-attack] wrote {manifest['n_output_files']} files -> {manifest['out_dir']}")
    print(f"[redteam-attack] manifest -> {manifest['out_dir']}/manifest.json")
    if manifest["score_stats"] is not None:
        stats = manifest["score_stats"]
        print(f"[redteam-attack] score max={stats['max']:.4f} "
              f"mean={stats['mean']:.4f} "
              f"above_threshold={stats['n_above_threshold']}")


if __name__ == "__main__":
    main()
