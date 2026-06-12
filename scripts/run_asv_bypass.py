"""CLI: end-to-end ASV-bypass evaluation.

Chains the four ASV-bypass pieces into one command:

    trial protocol  ->  ASV calibration  ->  tandem (ASV + CM)  ->  attack×defense report
    common.trial_protocol   attack.verify.calibrate   pipeline.tandem   pipeline.report

Two modes:

  --demo  (default when models/data are not fully supplied)
      Fully synthetic, deterministic dummy scorers. No models, no audio files,
      no GPU — useful for CI and for seeing the whole pipeline produce its four
      output artifacts. Builds a small attack×defense grid.

  real    (give --speaker-data --cloned-dir --verifier-ckpt --cm-ckpt)
      Loads a SpeakerVerifier (ECAPA-TDNN) and an alt detector (LCNN/RawNet2)
      and evaluates a real submission.

Outputs (under --output-dir):
    protocol.csv        the trial protocol (TrialSet)
    calibration.json    ASV operating point (EER threshold)
    tandem.json         per (attack, defense) tandem results
    results/asv_mapping_report.md   the attack×defense ASR matrix

By construction ``asr_tandem <= asr_asv`` always holds (tandem bypass requires
ASV-accept AND CM-evade, a subset of ASV-accept).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..common.trial_protocol import build_trial_protocol
from ..attack.verify.calibrate import calibrate_asv
from ..pipeline.tandem import evaluate_tandem
from ..pipeline.report import generate_asv_mapping_report


# --------------------------------------------------------------------------- #
# Synthetic demo: deterministic, model-free scorers keyed off file-path tokens.
# --------------------------------------------------------------------------- #
DEMO_ATTACKS = {        # name -> ASV-evade skill (how victim-like the clone is)
    "xtts": 0.62,
    "rvc": 0.50,
    "xtts+codec": 0.70,
}
DEMO_DEFENSES = {       # name -> CM catch skill (how well it flags spoofs)
    "LCNN": 0.55,
    "RawNet2": 0.60,
    "AASIST": 0.78,
}
DEMO_CM_THRESHOLD = 0.5


def _u(s: str) -> float:
    """Deterministic value in [0, 1) from a string (process-independent)."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _spk(path: str):
    for part in Path(path).parts:
        if part.startswith("spk"):
            return part
    return None


def _is_clone(path: str) -> bool:
    return "/clone/" in path.replace("\\", "/")


def _attack_of(path: str):
    parts = path.replace("\\", "/").split("/")
    if "clone" in parts:
        i = parts.index("clone")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def _demo_asv(attack_evade: float):
    """similarity(enroll, test): genuine high, impostor low, clone ~ attack skill."""
    def fn(enroll: str, test: str) -> float:
        if _is_clone(test):
            return 0.30 + 0.45 * attack_evade + 0.15 * (_u(test) - 0.5)
        same = _spk(enroll) is not None and _spk(enroll) == _spk(test)
        base = 0.80 if same else 0.20
        return base + 0.15 * (_u(enroll + test) - 0.5)
    return fn


def _demo_cm(catch: float):
    """spoofness(test): rises with defense skill, falls with attack stealth;
    >= threshold == detected. Stealthier attacks evade weaker defenses."""
    def fn(test: str) -> float:
        evade = DEMO_ATTACKS.get(_attack_of(test), 0.5)
        return 0.50 + 0.40 * catch - 0.40 * evade + 0.15 * (_u("cm:" + test) - 0.5)
    return fn


def _demo_speakers(n_speakers=6, n_utt=4):
    return {f"spk{i}": [f"data/demo/spk{i}/utt{j}.wav" for j in range(n_utt)]
            for i in range(n_speakers)}


def _demo_clones(speakers, attack, n_per_speaker=3):
    return {s: [f"data/demo/clone/{attack}/{s}_{k}.wav" for k in range(n_per_speaker)]
            for s in speakers}


def run_demo(output_dir: str, seed: int = 42) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    speakers = _demo_speakers()

    entries = []
    calib = None
    base_set = None
    tandem_dump = []

    for attack, evade in DEMO_ATTACKS.items():
        ts = build_trial_protocol(speakers, _demo_clones(speakers, attack), seed=seed)
        if calib is None:
            # ASV scoring for genuine/impostor is attack-independent, so calibrate once.
            calib = calibrate_asv(ts, _demo_asv(evade))
            base_set = ts
        spoof_trials = ts.spoof()
        for dname, catch in DEMO_DEFENSES.items():
            res = evaluate_tandem(spoof_trials, _demo_asv(evade), calib.threshold,
                                  _demo_cm(catch), DEMO_CM_THRESHOLD)
            entries.append({"attack": attack, "defense": dname, "result": res})
            tandem_dump.append({"attack": attack, "defense": dname, **res.to_dict()})

    # artifacts
    base_set.to_csv(out / "protocol.csv")
    calib.save(out / "calibration.json")
    with open(out / "tandem.json", "w", encoding="utf-8") as f:
        json.dump(tandem_dump, f, indent=2)
    report_path = generate_asv_mapping_report(entries, calib, output_root=output_dir)

    print(f"[asv-bypass] demo: {len(speakers)} speakers, "
          f"{len(DEMO_ATTACKS)} attacks × {len(DEMO_DEFENSES)} defenses")
    print(f"[asv-bypass] ASV EER threshold = {calib.threshold:.4f} (EER={calib.eer:.4f})")
    print(f"[asv-bypass] protocol     -> {out / 'protocol.csv'}")
    print(f"[asv-bypass] calibration  -> {out / 'calibration.json'}")
    print(f"[asv-bypass] tandem        -> {out / 'tandem.json'}")
    print(f"[asv-bypass] report        -> {report_path}")
    return {"calibration": calib, "entries": entries, "report": report_path}


# --------------------------------------------------------------------------- #
# Real mode: load actual ECAPA verifier + alt detector.
# --------------------------------------------------------------------------- #
def run_real(args) -> dict:
    from glob import glob
    from ..attack.verify.speaker_verifier import SpeakerVerifier
    from ..defense.alt.detector import load_alt_detector
    from ..pipeline.tandem import cm_score_fn_from_detector

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.speaker_data, encoding="utf-8") as f:
        speaker_utterances = json.load(f)          # {speaker: [utt_wav, ...]}

    cloned_by_speaker = {}
    cdir = Path(args.cloned_dir)
    for spk in speaker_utterances:
        wavs = sorted(glob(str(cdir / spk / "*.wav")))
        if wavs:
            cloned_by_speaker[spk] = wavs

    ts = build_trial_protocol(speaker_utterances, cloned_by_speaker, seed=args.seed)

    verifier = SpeakerVerifier(args.verifier_ckpt) if args.verifier_ckpt else SpeakerVerifier()
    calib = calibrate_asv(ts, verifier.similarity)

    model, mtype = load_alt_detector(args.cm_ckpt, device=args.device or "cpu")
    cm_fn = cm_score_fn_from_detector(model, mtype, device=args.device or "cpu")

    res = evaluate_tandem(ts.spoof(), verifier.similarity, calib.threshold,
                          cm_fn, args.cm_threshold)
    entries = [{"attack": args.attack_name, "defense": args.cm_name, "result": res}]

    ts.to_csv(out / "protocol.csv")
    calib.save(out / "calibration.json")
    res.save(out / "tandem.json")
    report_path = generate_asv_mapping_report(entries, calib, output_root=args.output_dir)

    print(f"[asv-bypass] ASV EER threshold = {calib.threshold:.4f} (EER={calib.eer:.4f})")
    print(f"[asv-bypass] asr_asv={res.asr_asv:.4f}  asr_tandem={res.asr_tandem:.4f}")
    print(f"[asv-bypass] report -> {report_path}")
    return {"calibration": calib, "entries": entries, "report": report_path}


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end ASV-bypass evaluation.")
    p.add_argument("--demo", action="store_true",
                   help="Run fully synthetic (no models/audio). Default if data/ckpts are missing.")
    p.add_argument("--speaker-data", help="JSON: {speaker_id: [utterance_wav, ...]}.")
    p.add_argument("--cloned-dir", help="Directory of cloned wavs, with per-speaker subdirs.")
    p.add_argument("--verifier-ckpt", help="ECAPA-TDNN checkpoint (SpeechBrain). Optional.")
    p.add_argument("--cm-ckpt", help="Alt detector (LCNN/RawNet2) checkpoint.")
    p.add_argument("--cm-name", default="CM", help="Defense name shown in the report.")
    p.add_argument("--attack-name", default="submitted", help="Attack name shown in the report.")
    p.add_argument("--cm-threshold", type=float, default=DEMO_CM_THRESHOLD)
    p.add_argument("--output-dir", default="outputs/asv_bypass_eval")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    real_ready = args.speaker_data and args.cloned_dir and args.cm_ckpt
    if args.demo or not real_ready:
        if not args.demo and not real_ready:
            print("[asv-bypass] models/data not fully supplied — running synthetic demo. "
                  "(Provide --speaker-data --cloned-dir --cm-ckpt for a real run.)")
        run_demo(args.output_dir, args.seed)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
