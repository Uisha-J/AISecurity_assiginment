"""Dataset downloader.

Handles each dataset according to its access policy:
  - PUBLIC  : direct URL via HTTP, auto-extract
  - HUGGING : Hugging Face hub (anonymous OK for public repos)
  - GATED   : printed instructions, user must register manually

Usage:
    python -m voice_defense.scripts.download_data --datasets librispeech_devclean
    python -m voice_defense.scripts.download_data --datasets musan rirs_noises
    python -m voice_defense.scripts.download_data --list
"""

from __future__ import annotations
import argparse
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from urllib.request import urlopen
import shutil


DATA_ROOT = Path("voice_defense/data")


@dataclass
class Dataset:
    name: str
    url: Optional[str]
    target_dir: Path
    size_mb: int
    access: str                       # "public" | "gated"
    extract: Optional[str] = None     # "tar.gz" | "zip" | None
    note: str = ""
    instructions: str = ""


DATASETS: dict[str, Dataset] = {
    # ============================================================ Bonafide
    "librispeech_devclean": Dataset(
        name="LibriSpeech dev-clean",
        url="https://www.openslr.org/resources/12/dev-clean.tar.gz",
        target_dir=DATA_ROOT / "bonafide" / "librispeech_devclean",
        size_mb=337,
        access="public",
        extract="tar.gz",
        note="Small bonafide test set. Best starter.",
    ),
    "librispeech_train100": Dataset(
        name="LibriSpeech train-clean-100",
        url="https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        target_dir=DATA_ROOT / "bonafide" / "librispeech_train100",
        size_mb=6300,
        access="public",
        extract="tar.gz",
        note="100h clean speech. Main bonafide training corpus (free tier).",
    ),

    # ============================================================ Augmentation
    "musan": Dataset(
        name="MUSAN",
        url="https://www.openslr.org/resources/17/musan.tar.gz",
        target_dir=DATA_ROOT / "augment" / "musan",
        size_mb=11000,
        access="public",
        extract="tar.gz",
        note="Noise/music/speech for additive augmentation.",
    ),
    "rirs_noises": Dataset(
        name="RIRS_NOISES",
        url="https://www.openslr.org/resources/28/rirs_noises.zip",
        target_dir=DATA_ROOT / "augment" / "rirs",
        size_mb=14000,
        access="public",
        extract="zip",
        note="Real + simulated room impulse responses.",
    ),

    # ============================================================ Spoof / wild
    "mlaad": Dataset(
        name="MLAAD (multi-language TTS deepfakes)",
        url=None,
        target_dir=DATA_ROOT / "wild" / "mlaad",
        size_mb=50000,
        access="public",
        instructions=(
            "Via Hugging Face Hub:\n"
            "  pip install huggingface_hub\n"
            "  huggingface-cli download mueller91/MLAAD --repo-type dataset \\\n"
            "    --local-dir voice_defense/data/wild/mlaad\n"
        ),
    ),
    "in_the_wild": Dataset(
        name="In-the-Wild",
        url=None,
        target_dir=DATA_ROOT / "wild" / "in_the_wild",
        size_mb=7500,
        access="gated",
        instructions=(
            "Request access at https://deepfake-total.com/in_the_wild\n"
            "After email confirmation, place the extracted folder at:\n"
            "  voice_defense/data/wild/in_the_wild/\n"
        ),
    ),
    "asvspoof2019_la": Dataset(
        name="ASVspoof 2019 LA",
        url=None,
        target_dir=DATA_ROOT / "spoof_known" / "asvspoof2019_la",
        size_mb=25000,
        access="gated",
        instructions=(
            "Register at https://datashare.ed.ac.uk/handle/10283/3336\n"
            "Download LA.zip and extract to:\n"
            "  voice_defense/data/spoof_known/asvspoof2019_la/\n"
        ),
    ),
    "asvspoof5": Dataset(
        name="ASVspoof 5",
        url=None,
        target_dir=DATA_ROOT / "spoof_known" / "asvspoof5",
        size_mb=100000,
        access="gated",
        instructions=(
            "Register via Zenodo: https://zenodo.org/records/14498691\n"
            "Place extracted files at:\n"
            "  voice_defense/data/spoof_known/asvspoof5/\n"
        ),
    ),
    "voxceleb1": Dataset(
        name="VoxCeleb 1",
        url=None,
        target_dir=DATA_ROOT / "bonafide" / "voxceleb1",
        size_mb=33000,
        access="gated",
        instructions=(
            "Fill registration form at\n"
            "  https://www.robots.ox.ac.uk/~vgg/data/voxceleb/\n"
            "Wait for download link (usually <1 day).\n"
        ),
    ),
}


# -------------------------------------------------------------------- helpers
def _human_mb(n: int) -> str:
    return f"{n/1024:.1f} GB" if n >= 1024 else f"{n} MB"


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Streaming download with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] already on disk: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  [download] {url}\n      -> {dest}")
    with urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with dest.open("wb") as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = done / total * 100
                    print(f"\r      {done/1e6:.0f}/{total/1e6:.0f} MB  {pct:.1f}%",
                          end="", flush=True)
    print()


def _extract(archive: Path, target: Path, kind: str) -> None:
    target.mkdir(parents=True, exist_ok=True)
    print(f"  [extract] {archive.name} -> {target}")
    if kind == "tar.gz":
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(target)
    elif kind == "zip":
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(target)
    else:
        raise ValueError(f"unknown extract kind: {kind}")


def _list() -> None:
    print(f"{'KEY':<22} {'NAME':<35} {'SIZE':<10} {'ACCESS':<8}")
    print("-" * 75)
    for k, d in DATASETS.items():
        print(f"{k:<22} {d.name:<35} {_human_mb(d.size_mb):<10} {d.access:<8}")


def _fetch_one(key: str) -> None:
    d = DATASETS[key]
    print(f"\n=== {d.name} ({_human_mb(d.size_mb)}, {d.access}) ===")
    if d.access == "gated" or d.url is None:
        print("Manual access required.")
        print(d.instructions)
        return
    archive_dir = DATA_ROOT / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / Path(d.url).name
    try:
        _download(d.url, archive_path)
    except Exception as e:
        print(f"  [error] download failed: {e}")
        return
    if d.extract:
        try:
            _extract(archive_path, d.target_dir, d.extract)
        except Exception as e:
            print(f"  [error] extract failed: {e}")
            return
    print(f"  [done] -> {d.target_dir}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--datasets", nargs="*", default=[],
                   help="keys from --list output")
    p.add_argument("--all-public", action="store_true",
                   help="download every dataset with access=public")
    args = p.parse_args()

    if args.list or (not args.datasets and not args.all_public):
        _list()
        return 0

    keys = list(args.datasets)
    if args.all_public:
        keys += [k for k, d in DATASETS.items() if d.access == "public"]
    keys = list(dict.fromkeys(keys))  # dedupe, preserve order

    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        print(f"[err] unknown dataset keys: {unknown}")
        print("Use --list to see valid keys.")
        return 1

    for k in keys:
        _fetch_one(k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
