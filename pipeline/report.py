"""Generate simulation report: markdown + visualizations."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_simulation_report(report, output_root):
    """Generate comprehensive markdown report from simulation results."""
    results_dir = Path(output_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "simulation_report.md"

    lines = [
        "# Voice Deepfake Attack & Defense — Simulation Report\n",
        "## 1. Attack Results (Red Team)\n",
        "XTTS v2 음성 복제 → ECAPA-TDNN 화자 인증 우회\n",
    ]

    if "attack" in report:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in report["attack"].items():
            lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        lines.append("")

    lines.append("## 2. Defense Results (Blue Team)\n")

    if "defense_alt" in report:
        lines.append("### Alternative Detector (LCNN / RawNet2)\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in report["defense_alt"].items():
            lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        lines.append("")

    if "domain_gap" in report:
        lines.append("## 3. Domain Gap Analysis\n")
        lines.append("| Comparison | FID |")
        lines.append("|------------|-----|")
        gap = report["domain_gap"]
        if "fid_asvspoof_vs_cloned" in gap:
            lines.append(f"| ASVspoof vs XTTS Clone | {gap['fid_asvspoof_vs_cloned']:.2f} |")
        if "fid_asvspoof_vs_real" in gap:
            lines.append(f"| ASVspoof vs Real | {gap['fid_asvspoof_vs_real']:.2f} |")
        if "fid_real_vs_cloned" in gap:
            lines.append(f"| Real vs XTTS Clone | {gap['fid_real_vs_cloned']:.2f} |")
        lines.append("")

    lines.append("## 4. Conclusion\n")
    lines.append("공격 성공률(ASR)과 방어 탐지율을 비교하여 현재 방어 시스템의 한계와 개선 방향을 도출.\n")

    text = "\n".join(lines)
    report_path.write_text(text, encoding="utf-8")
    logger.info(f"Report saved: {report_path}")
    return str(report_path)


def _fmt_pct(x) -> str:
    return f"{100.0 * float(x):.1f}%"


def _as_dict(obj):
    return obj.to_dict() if hasattr(obj, "to_dict") else dict(obj)


def generate_asv_mapping_report(entries, calibration=None, output_root="./outputs",
                                filename="asv_mapping_report.md"):
    """Render an attack × defense ASR matrix from tandem (ASV+CM) evaluations.

    Args:
        entries: list of dicts, each
            ``{"attack": str, "defense": str, "result": TandemResult | dict}``.
            ``result`` must expose ``asr_asv`` and ``asr_tandem`` (a
            ``TandemResult`` or its ``to_dict()`` output).
        calibration: optional ``CalibrationResult`` | dict for the ASV operating
            point shown in the header.
        output_root: report is written to ``<output_root>/results/<filename>``.

    Returns:
        Path (str) to the written markdown file.

    Rows are attack vectors, columns are defenses; each cell is the tandem ASR
    (higher == attack wins). An extra "ASV 단독" column shows ASV-only bypass.
    The single most vulnerable (attack, defense) cell is highlighted with ★.
    """
    results_dir = Path(output_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / filename

    lines = ["# ASV-Bypass — Attack × Defense ASR Matrix\n"]

    if calibration is not None:
        c = _as_dict(calibration)
        thr = c.get("threshold_at_eer", c.get("threshold"))
        thr_s = f"{float(thr):.4f}" if thr is not None else "?"
        lines.append(
            f"ASV operating point calibrated at EER = {float(c.get('eer', 0.0)):.4f} "
            f"(threshold = {thr_s}); "
            f"genuine n={c.get('n_genuine', '?')}, impostor n={c.get('n_impostor', '?')}.\n"
        )

    if not entries:
        lines.append("_제공된 tandem 결과가 없습니다._\n")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"ASV mapping report saved (empty): {report_path}")
        return str(report_path)

    # Preserve insertion order of attacks / defenses.
    attacks, defenses = [], []
    tandem = {}    # (attack, defense) -> asr_tandem
    asv_only = {}  # attack -> asr_asv
    for e in entries:
        a, d = e["attack"], e["defense"]
        r = _as_dict(e["result"])
        if a not in attacks:
            attacks.append(a)
        if d not in defenses:
            defenses.append(d)
        tandem[(a, d)] = float(r.get("asr_tandem", 0.0))
        asv_only.setdefault(a, float(r.get("asr_asv", 0.0)))

    worst_cell = max(tandem.items(), key=lambda kv: kv[1])[0]

    lines.append("셀 = tandem ASR(%) — 높을수록 공격 성공(방어 실패). ★ = 가장 취약한 조합.\n")
    lines.append("| 공격 \\ 방어 | ASV 단독 | " + " | ".join(defenses) + " |")
    lines.append("|" + "---|" * (len(defenses) + 2))
    for a in attacks:
        row = [a, _fmt_pct(asv_only.get(a, 0.0))]
        for d in defenses:
            if (a, d) in tandem:
                cell = _fmt_pct(tandem[(a, d)])
                if (a, d) == worst_cell:
                    cell = f"**{cell} ★**"
            else:
                cell = "—"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    atk_mean = {a: _mean([tandem[(a, d)] for d in defenses if (a, d) in tandem]) for a in attacks}
    def_mean = {d: _mean([tandem[(a, d)] for a in attacks if (a, d) in tandem]) for d in defenses}
    strongest_attack = max(atk_mean, key=atk_mean.get)
    strongest_defense = min(def_mean, key=def_mean.get)

    lines.append("## 요약\n")
    lines.append(f"- **가장 취약한 조합**: 공격 `{worst_cell[0]}` × 방어 `{worst_cell[1]}` "
                 f"→ tandem ASR {_fmt_pct(tandem[worst_cell])}")
    lines.append(f"- **가장 강력한 공격**: `{strongest_attack}` "
                 f"(평균 tandem ASR {_fmt_pct(atk_mean[strongest_attack])})")
    lines.append(f"- **가장 강한 방어**: `{strongest_defense}` "
                 f"(평균 tandem ASR {_fmt_pct(def_mean[strongest_defense])})")
    sa_asv, sa_tandem = asv_only.get(strongest_attack, 0.0), atk_mean[strongest_attack]
    if sa_asv - sa_tandem > 1e-9:
        lines.append(f"- `{strongest_attack}`는 ASV를 {_fmt_pct(sa_asv)} 뚫지만 CM 결합 시 평균 "
                     f"{_fmt_pct(sa_tandem)}로 떨어짐 — CM이 방어에 기여함을 정량적으로 보여줌.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"ASV mapping report saved: {report_path}")
    return str(report_path)
