"""End-to-end simulation: Attack → Defense → Compare → Report.

Runs the full Red vs Blue experiment:
  1. Attack: clone voices + measure ASR against speaker verification
  2. Defense: detect clones with AASIST and LCNN/RawNet2
  3. Analyze: domain gap (FID + t-SNE), fine-tune comparison
  4. Report: generate markdown + visualizations
"""

import json
import logging
from glob import glob
from pathlib import Path

import yaml

from ..common.compat import apply_all_patches

logger = logging.getLogger(__name__)


def run_simulation(config_path="configs/simulation.yaml", data_root="./data", output_root="./outputs"):
    """Run the full attack-defense simulation."""
    apply_all_patches()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(output_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {}

    # ================================================================ PHASE 1: ATTACK
    logger.info("=" * 60)
    logger.info("PHASE 1: RED TEAM — VOICE CLONING ATTACK")
    logger.info("=" * 60)

    from ..attack.clone.clone_voices import clone_voices_experiment
    from ..attack.verify.verify_clones import verify_clones_experiment
    from ..attack.verify.asr_calculator import compute_asr, compute_asr_summary, save_asr_report

    atk_cfg = cfg.get("attack", {})
    clone_result = clone_voices_experiment(
        data_root=data_root, output_root=output_root,
        num_target_speakers=atk_cfg.get("num_target_speakers", 20),
        sample_durations=atk_cfg.get("sample_durations", [5, 10, 30]),
        texts=atk_cfg.get("texts"),
        tts_model=atk_cfg.get("tts_model", "tts_models/multilingual/multi-dataset/xtts_v2"),
        device=cfg.get("device", "cuda"),
    )

    vf_cfg = cfg.get("verification", {})
    scores_df = verify_clones_experiment(
        cloned_files=clone_result["cloned_files"],
        reference_samples=clone_result["reference_samples"],
        output_root=output_root,
        threshold_sweep=vf_cfg.get("threshold_sweep", [0.15, 0.20, 0.25, 0.30, 0.35]),
        device=cfg.get("device", "cuda"),
    )

    save_asr_report(scores_df, str(results_dir / "attack_summary.csv"))
    report["attack"] = compute_asr_summary(scores_df)

    # ================================================================ PHASE 2: DEFENSE
    logger.info("=" * 60)
    logger.info("PHASE 2: BLUE TEAM — DEEPFAKE DETECTION")
    logger.info("=" * 60)

    cloned_paths = sorted(glob(f"{output_root}/cloned_audio/**/clone_*.wav", recursive=True))
    real_paths = sorted(glob(f"{output_root}/cloned_audio/**/ref_*.wav", recursive=True))

    def_cfg = cfg.get("defense", {})

    # 2a. Alternative detector (LCNN / RawNet2)
    if def_cfg.get("alt_detector", {}).get("enabled", True):
        asvspoof_root = str(Path(data_root) / "asvspoof2019")
        alt_cfg = def_cfg["alt_detector"]

        from ..defense.alt.train import train_alt_detector
        from ..defense.alt.detector import evaluate_on_clones

        logger.info("Training alternative detector...")
        train_result = train_alt_detector(
            asvspoof_root, output_root,
            model_type=alt_cfg.get("model_type", "lcnn"),
            epochs=alt_cfg.get("epochs", 30),
            device=cfg.get("device", "cuda"),
        )

        logger.info("Evaluating on cloned voices...")
        det_metrics = evaluate_on_clones(
            train_result["best_model_path"], cloned_paths, real_paths, output_root,
            device=cfg.get("device", "cuda"),
        )
        report["defense_alt"] = det_metrics

    # ================================================================ PHASE 3: ANALYSIS
    logger.info("=" * 60)
    logger.info("PHASE 3: DOMAIN GAP ANALYSIS")
    logger.info("=" * 60)

    if def_cfg.get("domain_gap", {}).get("enabled", True) and cloned_paths:
        from ..defense.domain_gap.analysis import compute_domain_gap

        # Get ASVspoof spoof paths
        protocol = Path(data_root) / "asvspoof2019" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
        spoof_paths = []
        if protocol.exists():
            with open(protocol) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts[4] == "spoof":
                        spoof_paths.append(str(Path(data_root) / "asvspoof2019" / "ASVspoof2019_LA_train" / "flac" / f"{parts[1]}.flac"))

        if spoof_paths:
            gap = compute_domain_gap(spoof_paths, cloned_paths, real_paths, output_root)
            report["domain_gap"] = {k: v for k, v in gap.items() if k.startswith("fid")}

    # ================================================================ PHASE 4: REPORT
    logger.info("=" * 60)
    logger.info("PHASE 4: REPORT GENERATION")
    logger.info("=" * 60)

    from ..pipeline.report import generate_simulation_report
    report_path = generate_simulation_report(report, output_root)

    # Save raw results
    with open(results_dir / "simulation_results.json", "w") as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (float, int)) else str(vv)
                       for kk, vv in v.items()} for k, v in report.items()}, f, indent=2)

    logger.info(f"Simulation complete. Report: {report_path}")
    return report
