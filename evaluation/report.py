"""Write a Markdown + JSON report from evaluate_checkpoint() output."""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path


def _md_table(headline_rows: list[dict]) -> str:
    lines = ["| Protocol | N | EER (%) | min-tDCF |",
             "|---|---:|---:|---:|"]
    for r in headline_rows:
        lines.append(
            f"| {r['protocol']} | {r['n']} | {r['eer']*100:.2f} | {r['min_tdcf']:.4f} |"
        )
    return "\n".join(lines)


def _md_per_attack(headline_rows: list[dict]) -> str:
    out: list[str] = []
    for r in headline_rows:
        out.append(f"### {r['protocol']}")
        out.append("")
        out.append("| Attack tag | N | FAR / FRR |")
        out.append("|---|---:|---:|")
        for tag, stats in r["per_attack"].items():
            rate_key = "far" if "far" in stats else "frr"
            rate = stats.get(rate_key, 0.0)
            out.append(f"| {tag} | {stats['n']} | {rate_key.upper()} {rate*100:.2f}% |")
        out.append("")
    return "\n".join(out)


def write_report(results: dict, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save full JSON (with raw scores) for later analysis
    raw_path = out_dir / f"raw-{ts}.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    headline = []
    for r in results["protocols"]:
        h = {k: v for k, v in r.items() if not k.startswith("_")}
        headline.append(h)

    md = []
    md.append(f"# Voice Defense Evaluation Report")
    md.append("")
    md.append(f"- Generated: {ts}")
    md.append(f"- Checkpoint: `{results['checkpoint']}`")
    md.append(f"- Device: `{results['device']}`")
    md.append("")
    md.append("## Headline metrics")
    md.append("")
    md.append(_md_table(headline))
    md.append("")
    md.append("## Per-attack breakdown")
    md.append("")
    md.append(_md_per_attack(headline))
    md.append("")
    md.append("## Honesty checklist")
    md.append("- [ ] Train and eval protocols share NO attack algorithm")
    md.append("- [ ] Threshold was selected on dev, not eval")
    md.append("- [ ] WILD protocol included (external real-world data)")
    md.append("- [ ] Red-team submissions evaluated separately and not used for tuning")

    md_path = out_dir / f"report-{ts}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return {"json": str(raw_path), "markdown": str(md_path)}
