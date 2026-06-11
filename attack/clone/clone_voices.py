"""Voice cloning experiment via attack zoo's XTTSAttack."""

import logging
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from ..zoo.tts.xtts import XTTSAttack
from ...common.redteam_data import LibriSpeechSpeakerDataset, prepare_reference_samples

logger = logging.getLogger(__name__)


def clone_voices_experiment(
    data_root, output_root, librispeech_subset="test-clean",
    num_target_speakers=20, sample_durations=None, texts=None,
    tts_model="tts_models/multilingual/multi-dataset/xtts_v2",
    language="en", device="cuda",
):
    if sample_durations is None:
        sample_durations = [5, 10, 30]
    if texts is None:
        texts = [
            "Please verify my identity for account access.",
            "The quick brown fox jumps over the lazy dog.",
            "This is a voice authentication test sample.",
        ]

    output_dir = Path(output_root) / "cloned_audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LibriSpeechSpeakerDataset(data_root, librispeech_subset)
    target_speakers = dataset.select_speakers(num_target_speakers)
    logger.info(f"Selected {len(target_speakers)} target speakers")

    reference_samples = {}
    for spk_id in tqdm(target_speakers, desc="Preparing references"):
        refs = prepare_reference_samples(
            dataset.get_utterances(spk_id), str(output_dir / str(spk_id)),
            str(spk_id), durations=sample_durations,
        )
        reference_samples[spk_id] = refs

    generator = XTTSAttack(model_name=tts_model, language=language, device=device)
    cloned_files = []

    for spk_id in tqdm(target_speakers, desc="Cloning voices"):
        for duration in sample_durations:
            ref_path = reference_samples[spk_id][duration]
            clone_dir = output_dir / str(spk_id)
            clone_dir.mkdir(parents=True, exist_ok=True)
            for i, text in enumerate(texts):
                out_path = clone_dir / f"clone_{spk_id}_{duration}s_text{i:02d}.wav"
                result = generator.generate(text=text, reference_wav=ref_path)
                sf.write(str(out_path), result.waveform, result.sample_rate)
                cloned_files.append({
                    "speaker_id": spk_id, "duration": duration, "text_id": i,
                    "reference_path": ref_path, "cloned_path": str(out_path),
                })

    logger.info(f"Generated {len(cloned_files)} cloned audio files.")
    return {
        "target_speakers": target_speakers, "reference_samples": reference_samples,
        "cloned_files": cloned_files, "texts": texts, "sample_durations": sample_durations,
    }
