"""Train a CM (LCNN/RawNet2) on real vs cloned audio — no ASVspoof needed.

For the scoped scenario "ASV is bypassed -> CM catches some clones", the
countermeasure can be trained directly on:
    bonafide = real speaker audio (LibriSpeech references, ref_*.wav)
    spoof    = XTTS clones (clone_*.wav)
both produced by the attack/clone phase under <output-root>/cloned_audio/.

Usage:
    # after the attack phase has populated outputs/cloned_audio/
    python -m voice_defense.scripts.train_cm_on_clones \
        --output-root ./outputs --model-type lcnn --epochs 20 --device cuda

    # then feed the checkpoint into the tandem evaluation:
    python -m voice_defense.scripts.run_asv_bypass \
        --speaker-data data/speaker_data.json --cloned-dir outputs/cloned_audio \
        --cm-ckpt outputs/models/lcnn_best.pth --device cuda
"""

import argparse
import logging
from glob import glob

from ..common.compat import apply_all_patches
apply_all_patches()

from ..defense.alt.train import train_cm_on_clones

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Train CM on real vs cloned audio (no ASVspoof)")
    p.add_argument("--output-root", default="./outputs",
                   help="Root that contains cloned_audio/ (and where models/ is written)")
    p.add_argument("--cloned-glob", default=None,
                   help="Override glob for cloned wavs (default: <output-root>/cloned_audio/**/clone_*.wav)")
    p.add_argument("--real-glob", default=None,
                   help="Override glob for real wavs (default: <output-root>/cloned_audio/**/ref_*.wav)")
    p.add_argument("--model-type", default="lcnn", choices=["lcnn", "rawnet2"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cloned_glob = args.cloned_glob or f"{args.output_root}/cloned_audio/**/clone_*.wav"
    real_glob = args.real_glob or f"{args.output_root}/cloned_audio/**/ref_*.wav"
    cloned_paths = sorted(glob(cloned_glob, recursive=True))
    real_paths = sorted(glob(real_glob, recursive=True))

    logger.info("Found %d real (bonafide) + %d cloned (spoof) wavs", len(real_paths), len(cloned_paths))
    if not real_paths or not cloned_paths:
        logger.error("Missing audio. Run the attack/clone phase first "
                     "(e.g. run_simulation or run_clone_attack) so %s/cloned_audio is populated.",
                     args.output_root)
        return

    result = train_cm_on_clones(
        real_paths=real_paths, cloned_paths=cloned_paths,
        output_root=args.output_root, model_type=args.model_type,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, device=args.device,
    )
    logger.info("=" * 60)
    logger.info("CM 학습 완료: best dev EER=%.4f", result["best_eer"])
    logger.info("체크포인트: %s", result["best_model_path"])
    logger.info("→ run_asv_bypass --cm-ckpt %s 로 tandem 평가에 사용", result["best_model_path"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
