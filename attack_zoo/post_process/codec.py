"""Codec post-processing: re-encode through narrowband / lossy codecs.

Simulates the audio passing through phone networks (G.711), VoIP (Opus),
or music streaming (MP3/AAC). This is the single highest-impact augmentation
for telephony / voice-phishing scenarios.
"""

from __future__ import annotations
import io
import random
from typing import Optional, Sequence

import numpy as np

from ..base import PostProcessor


class CodecPostProcessor(PostProcessor):
    """Re-encode and decode through one of several codecs."""

    name = "codec"

    def __init__(
        self,
        codecs: Sequence[str] = ("opus", "mp3", "g711_alaw", "g711_ulaw"),
        bitrates: Optional[dict[str, int]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.codecs = list(codecs)
        self.bitrates = bitrates or {"opus": 16000, "mp3": 32000}
        self.rng = rng or random.Random()

    def is_available(self) -> bool:
        try:
            import torchaudio  # noqa: F401
            return True
        except ImportError:
            return False

    def apply(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch
        import torchaudio

        codec = self.rng.choice(self.codecs)
        wav_t = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
        sr = sample_rate

        # G.711 needs 8 kHz mono
        if codec in ("g711_alaw", "g711_ulaw"):
            if sr != 8000:
                wav_t = torchaudio.functional.resample(wav_t, sr, 8000)
                sr = 8000
            encoding = "ALAW" if codec.endswith("alaw") else "ULAW"
            out = self._roundtrip(wav_t, sr, format="wav", encoding=encoding, bits_per_sample=8)
        elif codec == "opus":
            # Opus operates well at 16k/24k/48k. Force 16k for telephony.
            if sr != 16000:
                wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
                sr = 16000
            out = self._roundtrip(wav_t, sr, format="ogg", encoding="OPUS",
                                  compression=self.bitrates.get("opus", 16000))
        elif codec == "mp3":
            out = self._roundtrip(wav_t, sr, format="mp3",
                                  compression=self.bitrates.get("mp3", 32000))
        else:
            out = wav_t  # unknown codec → passthrough

        # Resample back to original sr so downstream stays consistent
        if sr != sample_rate:
            out = torchaudio.functional.resample(out, sr, sample_rate)
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def _roundtrip(wav_t, sr: int, **save_kwargs):
        import torch, torchaudio
        buf = io.BytesIO()
        try:
            torchaudio.save(buf, wav_t, sr, **save_kwargs)
        except Exception:
            # backend lacks this codec — passthrough rather than crash
            return wav_t
        buf.seek(0)
        out, out_sr = torchaudio.load(buf)
        assert out_sr == sr or out_sr == 0  # some backends return 0; trust sr
        return out
