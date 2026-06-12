"""Preflight check before a real (Tier A) run.

Validates the environment + data layout so you fail in SECONDS, not hours into
a run. Run from the repo root on the remote GPU box:

    python -m voice_defense.scripts.preflight --data-root ./data

[OK]   = good
[WARN] = run still works but degraded / a phase will be skipped
[FAIL] = will break the run; fix before launching
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def _check(cond: bool, label: str, detail: str = "", warn: bool = False) -> bool:
    tag = "OK  " if cond else ("WARN" if warn else "FAIL")
    print(f"[{tag}] {label}" + (f"  ({detail})" if detail else ""))
    return cond or warn  # warnings do not fail the overall result


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight env + data check for a real run.")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--subset", default="test-clean")
    args = ap.parse_args()

    ok = True
    print("== environment ==")
    ok &= _check(sys.version_info[:2] >= (3, 10), "Python >= 3.10",
                 f"got {sys.version_info.major}.{sys.version_info.minor}")

    for mod in ["numpy", "scipy", "sklearn", "yaml", "soundfile", "librosa"]:
        try:
            importlib.import_module(mod)
            _check(True, f"import {mod}")
        except Exception as e:  # noqa: BLE001
            ok &= _check(False, f"import {mod}", str(e)[:60])

    try:
        import torch
        _check(True, "import torch", torch.__version__)
        cuda = torch.cuda.is_available()
        _check(cuda, "CUDA available",
               torch.cuda.get_device_name(0) if cuda else "no GPU -> XTTS/training very slow",
               warn=not cuda)
        if cuda:
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            _check(vram >= 6.0, "GPU VRAM >= 6 GB (XTTS needs ~6)", f"{vram:.1f} GB", warn=vram < 6.0)
        import torchaudio
        _check(True, "import torchaudio", torchaudio.__version__)
    except Exception as e:  # noqa: BLE001
        ok &= _check(False, "torch / torchaudio", str(e)[:60])

    print("\n== optional model deps (needed for a REAL run) ==")
    for mod, why in [("TTS", "XTTS voice cloning"), ("speechbrain", "ECAPA-TDNN ASR")]:
        try:
            importlib.import_module(mod)
            _check(True, f"import {mod}", why)
        except Exception:  # noqa: BLE001
            _check(False, f"import {mod}", f"{why}; pip install {mod}", warn=True)

    print("\n== data layout (--data-root) ==")
    dr = Path(args.data_root)
    ls = dr / "librispeech" / "LibriSpeech" / args.subset
    ls_ok = ls.is_dir() and any(ls.iterdir())
    ok &= _check(ls_ok, f"LibriSpeech {args.subset} (attack source)", str(ls))
    if ls_ok:
        n = sum(1 for d in ls.iterdir() if d.is_dir())
        _check(n >= 20, "LibriSpeech speaker dirs >= 20", f"{n} (else lower num_target_speakers)", warn=n < 20)

    asv_proto = dr / "asvspoof2019" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    asv_flac = dr / "asvspoof2019" / "ASVspoof2019_LA_train" / "flac"
    asv_ok = asv_proto.exists() and asv_flac.is_dir()
    _check(asv_ok, "ASVspoof2019 LA (defense + domain-gap)",
           str(dr / "asvspoof2019") if asv_ok else "missing -> defense phases will be SKIPPED", warn=True)

    print()
    if ok:
        msg = "ALL CRITICAL CHECKS PASSED."
        if not asv_ok:
            msg += " (ASVspoof missing: attack ASR will run, but LCNN defense + domain-gap are skipped.)"
        print(msg)
        return 0
    print("FIX THE [FAIL] ITEMS ABOVE BEFORE RUNNING.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
