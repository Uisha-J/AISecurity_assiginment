"""Speaker verification using SpeechBrain ECAPA-TDNN (192-dim embeddings)."""

import numpy as np
import torch

try:
    from speechbrain.inference.speaker import SpeakerRecognition
except ImportError:
    from speechbrain.pretrained import SpeakerRecognition


class SpeakerVerifier:
    def __init__(self, model_source="speechbrain/spkrec-ecapa-voxceleb",
                 save_dir="pretrained_models/spkrec-ecapa-voxceleb", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = SpeakerRecognition.from_hparams(
            source=model_source, savedir=save_dir, run_opts={"device": self.device},
        )

    def similarity(self, path_a, path_b):
        score, _ = self.model.verify_files(path_a, path_b)
        return score.item()

    def verify(self, path_a, path_b, threshold=0.25):
        score = self.similarity(path_a, path_b)
        return score >= threshold, score

    def batch_verify(self, enrollment_path, test_paths, threshold=0.25):
        return [
            {"path": p, "score": (s := self.similarity(enrollment_path, p)), "verified": s >= threshold}
            for p in test_paths
        ]
