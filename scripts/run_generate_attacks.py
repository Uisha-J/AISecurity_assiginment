"""CLI: generate a batch of spoofed audio using the Attack Zoo.

Example:
    python -m voice_defense.scripts.generate_attacks \
        --prompts-jsonl data/prompts/train.jsonl \
        --out-dir data/spoof_self/train \
        --pipelines xtts,rvc,openvoice \
        --n-samples 5000
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from ..attack_zoo.base import DummyTTS
from ..attack_zoo.orchestrator import AttackOrchestrator
from ..attack_zoo.post_process.codec import CodecPostProcessor
from ..attack_zoo.post_process.room_ir import RoomIRPostProcessor
from ..attack_zoo.post_process.noise_mix import NoisePostProcessor


GENERATOR_REGISTRY = {
    "synthetic_tts": ("voice_defense.attack_zoo.tts.synthetic_tts", "SyntheticTTSAttack"),
    "artifact_vc": ("voice_defense.attack_zoo.vc.artifact_vc", "ArtifactVCAttack"),
    "xtts": ("voice_defense.attack_zoo.tts.xtts", "XTTSAttack"),
    "openvoice": ("voice_defense.attack_zoo.tts.openvoice", "OpenVoiceAttack"),
    "tortoise": ("voice_defense.attack_zoo.tts.tortoise", "TortoiseAttack"),
    "bark": ("voice_defense.attack_zoo.tts.bark", "BarkAttack"),
    "rvc": ("voice_defense.attack_zoo.vc.rvc", "RVCAttack"),
    "seed_vc": ("voice_defense.attack_zoo.vc.seed_vc", "SeedVCAttack"),
    "dummy": (None, None),       # always-available fallback
}


def _load_generator(name: str, kwargs: dict):
    if name == "dummy":
        return DummyTTS()
    mod_name, cls_name = GENERATOR_REGISTRY[name]
    import importlib
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(**kwargs)


def _load_prompts(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"prompts file not found: {jsonl_path}. "
            f"create it with one JSON object per line, e.g. "
            f'{{"text": "hello world"}}'
        )
    # utf-8-sig transparently strips a BOM if the file was saved by
    # Notepad / PowerShell `>` redirection on Windows.
    prompts: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8-sig") as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                prompts.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{jsonl_path}:{ln}: invalid JSON ({e.msg}). "
                    f"each line must be a single JSON object."
                ) from e
    if not prompts:
        raise ValueError(f"{jsonl_path} contains no prompts.")
    return prompts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts-jsonl", required=True,
                   help="JSONL with keys: text, reference_wav, target_wav (any subset)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pipelines", default="dummy",
                   help="comma-separated names from registry")
    p.add_argument("--post-process", default="codec,noise",
                   help="comma-separated post-processors")
    p.add_argument("--rir-dir", default="data/augment/rirs")
    p.add_argument("--noise-dir", default="data/augment/musan")
    p.add_argument("--n-samples", type=int, default=0,
                   help="Total samples to write; 0 = one per prompt")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--gen-kwargs-json", default="{}",
                   help='JSON object: {"xtts": {"language":"en"}, "rvc": {"model_pth":"..."}}')
    args = p.parse_args()

    rng = random.Random(args.seed)
    gen_kwargs_all = json.loads(args.gen_kwargs_json)
    gen_names = [s.strip() for s in args.pipelines.split(",") if s.strip()]
    generators = []
    for name in gen_names:
        kw = gen_kwargs_all.get(name, {})
        try:
            g = _load_generator(name, kw)
        except Exception as e:
            print(f"[skip] could not instantiate '{name}': {e}")
            continue
        if g.is_available():
            generators.append(g)
            print(f"[ok] generator: {g.name}")
        else:
            print(f"[skip] '{name}' not available (missing deps or weights)")
    if not generators:
        print("[fatal] no generators available. Falling back to dummy.")
        generators = [DummyTTS()]

    post: list = []
    requested = [s.strip() for s in args.post_process.split(",") if s.strip()]
    if "codec" in requested:
        c = CodecPostProcessor(rng=rng)
        if c.is_available():
            post.append(c)
    if "rir" in requested:
        r = RoomIRPostProcessor(args.rir_dir, rng=rng)
        if r.is_available():
            post.append(r)
    if "noise" in requested:
        n = NoisePostProcessor(args.noise_dir, rng=rng)
        if n.is_available():
            post.append(n)
    print(f"[ok] post-processors: {[p.name for p in post]}")

    orch = AttackOrchestrator(generators, post, rng=rng)

    prompts = _load_prompts(Path(args.prompts_jsonl))
    n = args.n_samples if args.n_samples > 0 else len(prompts)
    print(f"[run] generating {n} samples into {args.out_dir} "
          f"(prompts={len(prompts)}, seed={args.seed})")
    written = orch.generate_batch(prompts=prompts, out_dir=args.out_dir, n_total=n)
    n_failed = n - len(written)
    print(f"[done] wrote {len(written)}/{n} files  "
          f"(failed={n_failed})  manifest -> {args.out_dir}/manifest.json")
    if n_failed and len(written) == 0:
        print("[warn] every sample failed — check that prompts have the keys "
              "your selected pipelines require (text for TTS, target_wav for VC).")


if __name__ == "__main__":
    main()
