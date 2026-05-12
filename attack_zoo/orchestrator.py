"""Attack orchestrator: random pipeline composition for diverse spoof generation.

Picks one (generator, post-processing chain) per sample, runs it, and writes
out (wav, metadata.json) pairs. Built for *throughput diversity* — the more
unlike combinations you stuff into training, the better the defender generalizes.
"""

from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import soundfile as sf

from .base import AttackGenerator, PostProcessor


@dataclass
class PipelineSpec:
    """One concrete attack pipeline."""
    generator: AttackGenerator
    post_chain: list[PostProcessor] = field(default_factory=list)

    def name(self) -> str:
        return self.generator.name + (
            ("+" + "+".join(p.name for p in self.post_chain))
            if self.post_chain else ""
        )


class AttackOrchestrator:
    def __init__(
        self,
        generators: Sequence[AttackGenerator],
        post_processors: Sequence[PostProcessor] = (),
        rng: Optional[random.Random] = None,
        max_post_per_sample: int = 2,
        post_apply_prob: float = 0.7,
    ) -> None:
        self.generators = [g for g in generators if g.is_available()]
        self.post_processors = [p for p in post_processors if p.is_available()]
        if not self.generators:
            raise RuntimeError(
                "No attack generators available. Install at least one of "
                "TTS/openvoice/tortoise/bark/rvc-python/seed-vc."
            )
        self.rng = rng or random.Random()
        self.max_post_per_sample = max_post_per_sample
        self.post_apply_prob = post_apply_prob

    def sample_pipeline(self) -> PipelineSpec:
        gen = self.rng.choice(self.generators)
        post: list[PostProcessor] = []
        if self.post_processors and self.rng.random() < self.post_apply_prob:
            n = self.rng.randint(1, max(1, min(self.max_post_per_sample,
                                                len(self.post_processors))))
            post = self.rng.sample(self.post_processors, k=n)
        return PipelineSpec(generator=gen, post_chain=post)

    # ------------------------------------------------------------------ run one
    def run_one(
        self,
        *,
        text: Optional[str] = None,
        reference_wav: Optional[str] = None,
        target_wav: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, int, dict]:
        pipe = self.sample_pipeline()
        result = pipe.generator.generate(
            text=text, reference_wav=reference_wav,
            target_wav=target_wav, seed=seed,
        )
        wav, sr = result.waveform, result.sample_rate
        applied: list[str] = []
        for post in pipe.post_chain:
            try:
                wav = post.apply(wav, sr)
                applied.append(post.name)
            except Exception as e:
                # never crash a whole run because of a single processor
                applied.append(f"{post.name}:FAIL({type(e).__name__})")
        meta = result.metadata
        meta.post_processing = applied
        return wav, sr, json.loads(meta.to_json())

    # ------------------------------------------------------------------ batch
    def generate_batch(
        self,
        *,
        prompts: Iterable[dict],
        out_dir: str | Path,
        n_total: Optional[int] = None,
    ) -> list[Path]:
        """`prompts` is an iterable of dicts with keys: text, reference_wav, target_wav.

        We loop prompts (with replacement if exhausted) until n_total written
        or prompts iterable is exhausted (whichever comes first).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        prompts_list = list(prompts)
        if not prompts_list:
            raise ValueError("`prompts` must be non-empty")

        written: list[Path] = []
        target = n_total if n_total is not None else len(prompts_list)
        for i in range(target):
            p = prompts_list[i % len(prompts_list)]
            try:
                wav, sr, meta = self.run_one(
                    text=p.get("text"),
                    reference_wav=p.get("reference_wav"),
                    target_wav=p.get("target_wav"),
                    seed=p.get("seed"),
                )
            except Exception as e:
                # log + skip, keep batch alive
                err_path = out_dir / f"sample_{i:06d}.error.txt"
                err_path.write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
                continue
            wav_path = out_dir / f"sample_{i:06d}.wav"
            meta_path = out_dir / f"sample_{i:06d}.json"
            sf.write(str(wav_path), wav, sr)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
            written.append(wav_path)
        return written

    # ------------------------------------------------------------------ info
    def summary(self) -> dict:
        return {
            "generators_available": [g.name for g in self.generators],
            "post_processors_available": [p.name for p in self.post_processors],
        }
