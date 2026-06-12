"""Full cross-attack generalization MATRIX.

Generates clones from several attacks, then for EACH attack trains a CM on it
and measures detection rate on EVERY attack. The result is an NxN matrix:
rows = train-on, cols = test-on. The diagonal is "seen"; off-diagonal is the
generalization to unseen attacks.

Usage:
    python -m voice_defense.scripts.run_cross_attack_matrix \
        --attacks xtts,synthetic_tts,artifact_vc --speakers 20 --epochs 15 --device cuda
"""

import argparse
import json
import logging
from glob import glob
from pathlib import Path

import numpy as np

from ..common.compat import apply_all_patches
apply_all_patches()

from ..attack.clone.clone_multi import generate_clones_for_attacks
from ..defense.alt.train import train_cm_on_clones
from ..defense.alt.detector import load_alt_detector, detect_single
from ..evaluation.metrics import compute_eer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _spoof_scores(paths, model, mtype, device):
    return [detect_single(p, model, mtype, device=device)["spoof_score"] for p in paths]


def main():
    p = argparse.ArgumentParser(description="Cross-attack generalization matrix")
    p.add_argument("--attacks", default="xtts,synthetic_tts,artifact_vc")
    p.add_argument("--seen", default=None, help="restrict training to these (default: all attacks)")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-root", default="./outputs")
    p.add_argument("--speakers", type=int, default=20)
    p.add_argument("--ref-duration", type=int, default=10)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reuse-clones", action="store_true",
                   help="skip regeneration if clones already exist under output-root/cross_attack")
    args = p.parse_args()

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    base = Path(args.output_root) / "cross_attack"

    # --- generate (or reuse) clones ---
    if args.reuse_clones and base.exists():
        per = {}
        for a in attacks:
            clones = sorted(glob(f"{base}/{a}/**/clone_*.wav", recursive=True))
            refs = sorted(glob(f"{base}/{a}/**/ref_*.wav", recursive=True))
            if clones:
                per[a] = {"clones": clones, "refs": refs}
        logger.info("Reusing existing clones: %s", {a: len(d["clones"]) for a, d in per.items()})
    else:
        per = generate_clones_for_attacks(
            attack_names=attacks, data_root=args.data_root, output_root=args.output_root,
            num_target_speakers=args.speakers, ref_duration=args.ref_duration, device=args.device,
        )

    avail = list(per.keys())
    seen_list = [s.strip() for s in args.seen.split(",")] if args.seen else avail
    seen_list = [s for s in seen_list if s in avail]

    # --- build matrix ---
    matrix, diag_info = {}, {}
    for seen in seen_list:
        logger.info("=== Training CM on '%s' ===", seen)
        res = train_cm_on_clones(
            real_paths=per[seen]["refs"], cloned_paths=per[seen]["clones"],
            output_root=str(Path(args.output_root) / "_matrix"),
            epochs=args.epochs, device=args.device,
        )
        model, mtype = load_alt_detector(res["best_model_path"], args.device)
        ref_sc = _spoof_scores(per[seen]["refs"], model, mtype, args.device)
        seen_sc = _spoof_scores(per[seen]["clones"], model, mtype, args.device)
        _, thr = compute_eer(seen_sc + ref_sc, [1] * len(seen_sc) + [0] * len(ref_sc))
        diag_info[seen] = {"dev_eer": res["best_eer"], "cm_threshold": thr,
                           "frr_on_real": float(np.mean([s >= thr for s in ref_sc]))}
        matrix[seen] = {t: float(np.mean([s >= thr for s in _spoof_scores(per[t]["clones"], model, mtype, args.device)]))
                        for t in avail}

    # --- report ---
    results_dir = Path(args.output_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report = {"attacks": avail, "n_speakers": args.speakers,
              "n_clones_per_attack": {a: len(per[a]["clones"]) for a in avail},
              "matrix": matrix, "diag_info": diag_info}
    (results_dir / "cross_attack_matrix.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Cross-Attack Generalization Matrix\n",
             f"- 화자 수: {args.speakers}, 공격당 클론: {report['n_clones_per_attack']}",
             "- 값 = 탐지율(%). 행 = 학습 공격, 열 = 테스트 공격. 대각선 = seen.\n",
             "| train＼test | " + " | ".join(avail) + " |",
             "|---|" + "---|" * len(avail)]
    for seen in seen_list:
        cells = " | ".join(f"{matrix[seen][t]*100:.1f}%" + (" (seen)" if t == seen else "") for t in avail)
        lines.append(f"| **{seen}** | {cells} |")
    lines.append("\n## 해석\n")
    # XTTS column (modern neural clone) generalization
    if "xtts" in avail:
        off = [matrix[s]["xtts"] for s in seen_list if s != "xtts"]
        if off:
            lines.append(f"- 다른 공격으로 학습한 CM의 **XTTS 탐지율 평균 {np.mean(off)*100:.1f}%** "
                         f"(범위 {min(off)*100:.0f}~{max(off)*100:.0f}%) → 현대 신경망 복제의 미학습 탐지 한계.")
    md = "\n".join(lines)
    (results_dir / "cross_attack_matrix.md").write_text(md, encoding="utf-8")

    logger.info("=" * 60)
    print(md)
    logger.info("리포트: %s", results_dir / "cross_attack_matrix.md")


if __name__ == "__main__":
    main()
