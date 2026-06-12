"""One-shot data setup for a real (Tier A) benchmark run.

Places datasets into the SINGLE canonical layout that the pipeline actually
reads (`pipeline/simulate.py` + `common/redteam_data.py`), which is NOT the
same place `download_data.py` writes to. Run this once on the remote box, then
every command uses `--data-root ./data`.

What it does:
  1. Download + extract LibriSpeech (default `test-clean`) into
     `<data-root>/librispeech/LibriSpeech/<subset>/`  (the exact path
     `LibriSpeechSpeakerDataset` expects).
  2. Ensure `<data-root>/asvspoof2019/` exists and report whether the gated
     ASVspoof2019 LA tree is correctly placed (it cannot be auto-downloaded).
  3. Print a verification summary so you catch missing paths BEFORE the long run.

Usage:
    python -m voice_defense.scripts.setup_data                 # test-clean -> ./data
    python -m voice_defense.scripts.setup_data --subset dev-clean
    python -m voice_defense.scripts.setup_data --data-root ./data --skip-librispeech
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

LIBRISPEECH_URLS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
}

ASV_TRAIN_PROTO = "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
ASV_TRAIN_FLAC = "ASVspoof2019_LA_train/flac"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urlopen(url) as r:  # noqa: S310 (trusted openslr URL)
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with dest.open("wb") as f:
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    print(f"\r    {done/1e6:.0f}/{total/1e6:.0f} MB "
                          f"{done/total*100:.0f}%", end="", flush=True)
    print()


def setup_librispeech(data_root: Path, subset: str) -> None:
    if subset not in LIBRISPEECH_URLS:
        print(f"[setup] unknown subset '{subset}'. Choices: {list(LIBRISPEECH_URLS)}")
        return
    target = data_root / "librispeech" / "LibriSpeech" / subset
    if target.exists() and any(target.iterdir()):
        print(f"[setup] LibriSpeech {subset} already present at {target} — skipping.")
        return
    print(f"[setup] LibriSpeech {subset} -> {data_root / 'librispeech'}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{subset}.tar.gz"
        _download(LIBRISPEECH_URLS[subset], archive)
        print("  extracting ...")
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(data_root / "librispeech")
    if target.exists():
        n = sum(1 for _ in target.iterdir())
        print(f"[setup] OK: {n} speaker dirs at {target}")
    else:
        print(f"[setup] WARNING: expected {target} but it was not created.")


def check_asvspoof(data_root: Path) -> bool:
    root = data_root / "asvspoof2019"
    root.mkdir(parents=True, exist_ok=True)
    proto = root / ASV_TRAIN_PROTO
    flac = root / ASV_TRAIN_FLAC
    if proto.exists() and flac.exists():
        print(f"[setup] OK: ASVspoof2019 LA found at {root}")
        return True
    print(f"[setup] ASVspoof2019 LA NOT found at {root} (gated - manual download).")
    print("        1) Register + download LA.zip: https://datashare.ed.ac.uk/handle/10283/3336")
    print(f"        2) Extract so that these exist:")
    print(f"             {proto}")
    print(f"             {flac}/*.flac")
    print("        (Defense PHASE 2 / domain-gap are skipped until this is in place.)")
    return False


def main() -> int:
    p = argparse.ArgumentParser(description="Place datasets into the canonical ./data layout.")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--subset", default="test-clean", choices=list(LIBRISPEECH_URLS))
    p.add_argument("--skip-librispeech", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_librispeech:
        setup_librispeech(data_root, args.subset)
    asv_ok = check_asvspoof(data_root)

    print("\n[setup] summary:")
    ls = data_root / "librispeech" / "LibriSpeech" / args.subset
    print(f"  LibriSpeech {args.subset}: {'OK' if ls.exists() and any(ls.iterdir()) else 'MISSING'}  ({ls})")
    print(f"  ASVspoof2019 LA        : {'OK' if asv_ok else 'MISSING (gated, see above)'}")
    print(f"\n  Run with: --data-root {data_root}")
    return 0 if (args.skip_librispeech or (ls.exists() and any(ls.iterdir()))) else 1


if __name__ == "__main__":
    sys.exit(main())
