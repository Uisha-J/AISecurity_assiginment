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
