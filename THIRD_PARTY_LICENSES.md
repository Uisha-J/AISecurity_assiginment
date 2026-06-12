# Third-Party Licenses

This project orchestrates several external models and toolkits. They are **not**
bundled in this repository — you download them on demand. Each is governed by
its own license, which you must comply with. The table below is a convenience
summary; always check the upstream project for the authoritative terms.

## Attack zoo (voice cloning / conversion)

| Component | Project | License (as of upstream) |
|-----------|---------|--------------------------|
| XTTS v2 | Coqui TTS | Coqui Public Model License (CPML) — non-commercial; code MPL-2.0 |
| OpenVoice | MyShell OpenVoice | MIT (v1) / check upstream for v2 terms |
| Bark | Suno Bark | MIT (code); model weights for research use |
| Tortoise-TTS | neonbjb/tortoise-tts | Apache-2.0 |
| RVC | RVC-Project | MIT |
| seed-vc | Plachtaa/seed-vc | GPL-3.0 — check before redistribution |

## Defense / feature backbones

| Component | Project | License (as of upstream) |
|-----------|---------|--------------------------|
| WavLM / wav2vec2 / XLS-R | Microsoft / Meta (via HuggingFace) | MIT (WavLM) / MIT (wav2vec2) |
| AASIST | clovaai/aasist | MIT |
| ECAPA-TDNN | SpeechBrain | Apache-2.0 |
| SpeechBrain | speechbrain/speechbrain | Apache-2.0 |
| RawBoost (augmentation) | TakHemlata/RawBoost | research use; see paper |

## Datasets

| Dataset | License / Access |
|---------|------------------|
| LibriSpeech | CC BY 4.0 |
| ASVspoof 2019/2021 | ODC-By / challenge terms (registration) |
| MUSAN | CC BY 4.0 |
| RIRS_NOISES | Apache-2.0 |
| VoxCeleb | CC BY 4.0 (research) — gated download |
| In-the-Wild, MLAAD | research use; check upstream terms |

> Licenses change. Before any redistribution or non-research use, verify the
> current license of each component directly from its source.
