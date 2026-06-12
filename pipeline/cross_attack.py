"""Cross-attack generalization evaluation.

Story: train the countermeasure (CM) on ONE attack (``seen``), then test it on
attacks it was never trained on (``unseen``). A high detection rate on seen but
a low one on unseen is the quantitative evidence that "train == test" inflates
detection, and that generalization to new generators is the real challenge.

    train CM on  seen attack clones (+ real refs)
    evaluate on  every attack's clones separately
    report       detection_rate per attack, flagged seen vs unseen
"""

import json
import logging
from pathlib import Path

from ..defense.alt.train import train_cm_on_clones
from ..defense.alt.detector import load_alt_detector, detect_single
from ..evaluation.metrics import compute_eer

logger = logging.getLogger(__name__)


def _spoof_scores(paths, model, mtype, device):
    """spoof_score (= P(spoof)) for each clip; higher == more likely spoof."""
    return [detect_single(p, model, mtype, device=device)["spoof_score"] for p in paths]


def run_cross_attack_eval(
    per_attack,
    seen_attack,
    output_root,
    model_type="lcnn",
    epochs=20,
    cm_threshold=None,
    device="cuda",
):
    """per_attack: {attack: {"clones": [...], "refs": [...]}} from clone_multi.

    The CM decision threshold is CALIBRATED on the seen attack (real vs clone)
    rather than assumed at 0.5 — an EER-0 detector can still output spoof_scores
    below 0.5, so a fixed threshold would under-detect. Pass cm_threshold to override.

    Returns the report dict and writes JSON + Markdown under <output_root>/results/.
    """
    if seen_attack not in per_attack:
        raise ValueError(f"seen attack '{seen_attack}' not in generated attacks {list(per_attack)}")

    # --- train CM on the SEEN attack only ---
    seen = per_attack[seen_attack]
    logger.info("Training CM on SEEN attack '%s' (%d clones)...", seen_attack, len(seen["clones"]))
    train_res = train_cm_on_clones(
        real_paths=seen["refs"], cloned_paths=seen["clones"],
        output_root=output_root, model_type=model_type, epochs=epochs, device=device,
    )
    model, mtype = load_alt_detector(train_res["best_model_path"], device)

    # --- spoof scores per attack (and seen refs) ---
    attack_scores = {a: _spoof_scores(d["clones"], model, mtype, device) for a, d in per_attack.items()}
    ref_scores = _spoof_scores(seen["refs"], model, mtype, device)

    # --- calibrate CM threshold on SEEN (clones=spoof=1, refs=bonafide=0) ---
    if cm_threshold is None:
        seen_clone_scores = attack_scores[seen_attack]
        scores = seen_clone_scores + ref_scores
        labels = [1] * len(seen_clone_scores) + [0] * len(ref_scores)
        _, cm_threshold = compute_eer(scores, labels)
        logger.info("Calibrated CM threshold on seen attack: %.4f", cm_threshold)

    def _det_rate(scores):
        return (sum(s >= cm_threshold for s in scores) / len(scores)) if scores else None

    rows = [{
        "attack": a,
        "seen": a == seen_attack,
        "n_clones": len(attack_scores[a]),
        "detection_rate": _det_rate(attack_scores[a]),
    } for a in per_attack]
    frr = _det_rate(ref_scores)

    report = {
        "seen_attack": seen_attack,
        "model_type": model_type,
        "cm_threshold": cm_threshold,
        "train_dev_eer": train_res["best_eer"],
        "false_rejection_rate_on_real": frr,
        "per_attack": rows,
    }

    results_dir = Path(output_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "cross_attack.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _render_markdown(report)
    (results_dir / "cross_attack_report.md").write_text(md, encoding="utf-8")
    logger.info("Cross-attack report -> %s", results_dir / "cross_attack_report.md")
    return report


def _render_markdown(report):
    lines = [
        "# Cross-Attack Generalization Report\n",
        f"- **학습한 공격 (seen)**: `{report['seen_attack']}`",
        f"- **CM 모델**: {report['model_type']}",
        f"- **CM 임계값**: {report['cm_threshold']}",
        f"- **실음성 오탐률(FRR)**: {_fmt(report['false_rejection_rate_on_real'])}",
        "",
        "## 공격별 탐지율 (높을수록 잘 잡음)\n",
        "| 공격 | 구분 | 클론 수 | 탐지율 |",
        "|------|------|--------|--------|",
    ]
    for r in report["per_attack"]:
        tag = "**seen (학습)**" if r["seen"] else "unseen (미학습)"
        lines.append(f"| {r['attack']} | {tag} | {r['n_clones']} | {_fmt(r['detection_rate'])} |")

    seen_det = next((r["detection_rate"] for r in report["per_attack"] if r["seen"]), None)
    unseen = [r["detection_rate"] for r in report["per_attack"]
              if not r["seen"] and r["detection_rate"] is not None]
    lines.append("\n## 해석\n")
    if seen_det is not None and unseen:
        avg_unseen = sum(unseen) / len(unseen)
        lines.append(f"- seen 공격 탐지율 **{_fmt(seen_det)}** vs unseen 평균 **{_fmt(avg_unseen)}**")
        if seen_det - avg_unseen > 0.15:
            lines.append("- → 학습한 공격은 잘 잡지만 **처음 보는 공격은 탐지율이 떨어짐** "
                         "= 일반화의 한계가 정량적으로 드러남.")
        else:
            lines.append("- → unseen 공격에도 탐지율이 유지됨 = 어느 정도 일반화됨.")
    else:
        lines.append("- unseen 공격이 없어 일반화 비교 불가 (생성기를 2개 이상 설치하세요).")
    return "\n".join(lines)


def _fmt(x):
    return "N/A" if x is None else f"{x*100:.1f}%"
