"""Generate clones from MULTIPLE attack generators, kept separate per attack.

For cross-attack generalization: we need clones from different generators in
separate folders so a CM trained on one attack (``seen``) can be tested on the
others (``unseen``).

Layout produced:
    <output_root>/cross_attack/<attack_name>/<speaker>/ref_<spk>.wav     (bonafide)
    <output_root>/cross_attack/<attack_name>/<speaker>/clone_<spk>_*.wav (spoof)

Generators that are not installed are skipped with a warning (their adapter's
``is_available()`` returns False), so the harness still runs with whatever is
available — ``synthetic_tts`` and ``artifact_vc`` always work (no extra deps).
"""

import importlib
import logging
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from ...common.redteam_data import LibriSpeechSpeakerDataset, prepare_reference_samples

logger = logging.getLogger(__name__)

# name -> (module, class). TTS-family take (text, reference_wav); VC-family take
# (reference_wav, target_wav). We pass all, adapters ignore what they don't use.
GENERATOR_REGISTRY = {
    "xtts": ("voice_defense.attack.zoo.tts.xtts", "XTTSAttack"),
    "openvoice": ("voice_defense.attack.zoo.tts.openvoice", "OpenVoiceAttack"),
    "tortoise": ("voice_defense.attack.zoo.tts.tortoise", "TortoiseAttack"),
    "bark": ("voice_defense.attack.zoo.tts.bark", "BarkAttack"),
    "synthetic_tts": ("voice_defense.attack.zoo.tts.synthetic_tts", "SyntheticTTSAttack"),
    "artifact_vc": ("voice_defense.attack.zoo.vc.artifact_vc", "ArtifactVCAttack"),
    "rvc": ("voice_defense.attack.zoo.vc.rvc", "RVCAttack"),
    "seed_vc": ("voice_defense.attack.zoo.vc.seed_vc", "SeedVCAttack"),
}

DEFAULT_TEXTS = [
    "Please verify my identity for account access.",
    "The quick brown fox jumps over the lazy dog.",
    "This is a voice authentication test sample.",
]


def _load_generator(name, device="cuda", **kwargs):
    mod_name, cls_name = GENERATOR_REGISTRY[name]
    cls = getattr(importlib.import_module(mod_name), cls_name)
    try:
        return cls(device=device, **kwargs)
    except TypeError:
        return cls(**kwargs)  # adapters without a device arg


def _generate_one(gen, text, ref_path, seed):
    """Call a generator uniformly across TTS/VC families."""
    return gen.generate(text=text, reference_wav=ref_path, target_wav=ref_path, seed=seed)


def generate_clones_for_attacks(
    attack_names,
    data_root,
    output_root,
    librispeech_subset="test-clean",
    num_target_speakers=10,
    ref_duration=10,
    texts=None,
    device="cuda",
):
    """Clone the SAME speakers with each generator into per-attack folders.

    Returns: dict {attack_name: {"clones": [paths], "refs": [paths]}} for the
    generators that were actually available.
    """
    texts = texts or DEFAULT_TEXTS
    base = Path(output_root) / "cross_attack"
    base.mkdir(parents=True, exist_ok=True)

    dataset = LibriSpeechSpeakerDataset(data_root, librispeech_subset)
    speakers = dataset.select_speakers(num_target_speakers)
    logger.info("Cross-attack: %d speakers, attacks requested=%s", len(speakers), attack_names)

    results = {}
    for attack in attack_names:
        if attack not in GENERATOR_REGISTRY:
            logger.warning("Unknown generator '%s' — skipping.", attack)
            continue
        try:
            gen = _load_generator(attack, device=device)
        except Exception as e:
            logger.warning("Could not instantiate '%s' (%s) — skipping.", attack, e)
            continue
        if not gen.is_available():
            logger.warning("Generator '%s' not available (missing deps/weights) — skipping.", attack)
            continue

        attack_dir = base / attack
        clones, refs = [], []
        for spk in tqdm(speakers, desc=f"[{attack}] cloning"):
            spk_dir = attack_dir / str(spk)
            ref_map = prepare_reference_samples(
                dataset.get_utterances(spk), str(spk_dir), str(spk), durations=[ref_duration],
            )
            ref_path = ref_map[ref_duration]
            refs.append(ref_path)
            for i, text in enumerate(texts):
                out_path = spk_dir / f"clone_{spk}_text{i:02d}.wav"
                try:
                    res = _generate_one(gen, text, ref_path, seed=hash((attack, spk, i)) & 0xFFFF)
                    sf.write(str(out_path), res.waveform, res.sample_rate)
                    clones.append(str(out_path))
                except Exception as e:
                    logger.warning("  clone failed (%s, spk=%s, text=%d): %s", attack, spk, i, e)

        if clones:
            results[attack] = {"clones": clones, "refs": refs}
            logger.info("[%s] generated %d clones, %d refs", attack, len(clones), len(refs))

    if not results:
        raise RuntimeError(
            "No generators produced clones. Install at least one (e.g. `pip install TTS` "
            "for xtts) or rely on the always-available baselines (synthetic_tts, artifact_vc)."
        )
    return results
