"""Real ASV-bypass measurement on LibriSpeech speakers + XTTS clones.

Measures, with real models (ECAPA-TDNN + LCNN):
  - asr_asv     : how often XTTS clones pass speaker verification at its EER point
  - asr_tandem  : how often clones pass ASV AND evade the CM (the real bypass)

The ASV operating point is calibrated on genuine vs zero-effort-impostor trials
(not an arbitrary threshold), and the CM threshold is calibrated on real-vs-clone
so detection is measured at a meaningful point.

Usage (clones come from the cross-attack/clone phase):
    python -m voice_defense.scripts.run_asv_bypass_real \
        --clones-dir outputs/cross_attack/xtts --speakers 20 --device cuda
"""

import argparse
import json
import logging
from glob import glob
from pathlib import Path

import numpy as np

from ..common.compat import apply_all_patches
apply_all_patches()

from ..common.redteam_data import LibriSpeechSpeakerDataset
from ..common.trial_protocol import build_trial_protocol
from ..attack.verify.speaker_verifier import SpeakerVerifier
from ..attack.verify.calibrate import calibrate_asv
from ..defense.alt.train import train_cm_on_clones
from ..defense.alt.detector import load_alt_detector, detect_single
from ..pipeline.tandem import evaluate_tandem, cm_score_fn_from_detector
from ..evaluation.metrics import compute_eer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Real ASV-bypass measurement")
    p.add_argument("--clones-dir", default="outputs/cross_attack/xtts",
                   help="dir with per-speaker subdirs containing clone_*.wav and ref_*.wav")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--speakers", type=int, default=20)
    p.add_argument("--output-dir", default="outputs/asv_bypass_real")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cdir = Path(args.clones_dir)

    # --- speaker utterances (LibriSpeech) + XTTS clones, per speaker ---
    ds = LibriSpeechSpeakerDataset(args.data_root, "test-clean")
    speakers = [str(s) for s in ds.select_speakers(args.speakers)]
    speaker_utterances, cloned_by_speaker = {}, {}
    ref_paths, clone_paths = [], []
    for spk in speakers:
        utts = ds.get_utterances(int(spk))
        clones = sorted(glob(str(cdir / spk / "clone_*.wav")))
        refs = sorted(glob(str(cdir / spk / "ref_*.wav")))
        if utts and clones:
            speaker_utterances[spk] = utts
            cloned_by_speaker[spk] = clones
            clone_paths += clones
            ref_paths += refs
    logger.info("Speakers=%d, clones=%d, refs=%d", len(speaker_utterances), len(clone_paths), len(ref_paths))

    # --- trial protocol (genuine / impostor / spoof) ---
    ts = build_trial_protocol(speaker_utterances, cloned_by_speaker, seed=args.seed)
    logger.info("Trials: %s", ts.summary())

    # --- ASV: ECAPA-TDNN, calibrate threshold at EER (genuine vs impostor) ---
    logger.info("Loading ECAPA-TDNN speaker verifier (first run downloads ~80MB)...")
    verifier = SpeakerVerifier(device=args.device)
    calib = calibrate_asv(ts, verifier.similarity)
    logger.info("ASV EER=%.4f, threshold=%.4f", calib.eer, calib.threshold)

    # --- CM: train LCNN on real(refs) vs clone(XTTS), calibrate its threshold ---
    logger.info("Training CM (LCNN) on real vs XTTS clones...")
    cm_res = train_cm_on_clones(ref_paths, clone_paths, output_root=args.output_dir,
                                epochs=args.epochs, device=args.device)
    model, mtype = load_alt_detector(cm_res["best_model_path"], args.device)
    clone_sc = [detect_single(p, model, mtype, device=args.device)["spoof_score"] for p in clone_paths]
    ref_sc = [detect_single(p, model, mtype, device=args.device)["spoof_score"] for p in ref_paths]
    _, cm_threshold = compute_eer(clone_sc + ref_sc, [1] * len(clone_sc) + [0] * len(ref_sc))
    logger.info("CM dev EER=%.4f, calibrated CM threshold=%.4f", cm_res["best_eer"], cm_threshold)

    # --- tandem: ASV + CM jointly on spoof (clone) trials ---
    cm_fn = cm_score_fn_from_detector(model, mtype, device=args.device)
    res = evaluate_tandem(ts.spoof(), verifier.similarity, calib.threshold, cm_fn, cm_threshold)

    summary = {
        "n_speakers": len(speaker_utterances),
        "n_spoof_trials": res.n,
        "asv_eer": calib.eer,
        "asv_threshold": calib.threshold,
        "cm_threshold": cm_threshold,
        "asr_asv": res.asr_asv,            # clone passes ASV
        "asr_cm_evade": res.asr_cm_evade,  # clone evades CM
        "asr_tandem": res.asr_tandem,      # clone passes ASV AND evades CM
        "quadrants": res.counts,
    }
    (out / "asv_bypass_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ts.to_csv(out / "protocol.csv"); calib.save(out / "calibration.json")

    logger.info("=" * 60)
    logger.info("ASV 우회 실측 결과")
    logger.info("  ASV EER = %.1f%% (화자인증 자체 오류율)", calib.eer * 100)
    logger.info("  asr_asv    = %.1f%%  (복제가 화자인증을 통과한 비율)", res.asr_asv * 100)
    logger.info("  asr_tandem = %.1f%%  (ASV 통과 + CM 회피 = 완전 우회)", res.asr_tandem * 100)
    logger.info("  4사분면: %s", res.counts)
    logger.info("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
