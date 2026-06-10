# Voice Deepfake Attack & Defense Simulation

음성 딥페이크 **공격과 방어를 동시에 구현**하고,
모의 실험 파이프라인으로 공격 성공률(ASR)과 방어 탐지율을 자동 비교하는 프로젝트.

> 명령어 한 줄이면 공격 → 방어 → 분석 → 보고서까지 자동으로 돌아갑니다.

## 무엇을 하는 프로젝트인가요?

| | Red Team (공격) | Blue Team (방어) |
|---|---|---|
| **목표** | AI로 목소리를 복제해서 화자 인증 돌파 | 가짜 음성을 탐지해서 차단 |
| **핵심 기술** | XTTS v2 음성 복제 + ECAPA-TDNN 우회 | AASIST + LCNN + RawNet2 탐지기 |
| **측정 지표** | ASR (공격 성공률) | EER (탐지 오류율) |

이 프로젝트는 둘 중 하나만 하는 게 아니라,
**공격과 방어를 동등하게 만들어서 자동으로 붙여보고 결과를 리포트로 뽑아줍니다.**

## 구조

```
├── attack/                     # Red Team — 공격
│   ├── clone/                  # XTTS v2 음성 복제
│   ├── verify/                 # ECAPA-TDNN 화자 인증 우회 + ASR 측정
│   └── zoo/                    # 다중 공격 생성기 (XTTS, OpenVoice, Bark, RVC 등 6종)
│
├── defense/                    # Blue Team — 방어
│   ├── aasist/                 # WavLM + AASIST + OC-Softmax (1차 탐지기)
│   ├── alt/                    # LCNN + RawNet2 (대안 탐지기)
│   └── domain_gap/             # FID + t-SNE 도메인 갭 분석
│
├── pipeline/                   # 모의 실험 오케스트레이터
│   ├── simulate.py             # 공격 → 방어 → 분석 → 보고서 전체 흐름
│   └── report.py               # 산출물 생성
│
├── common/                     # 공용 유틸 (오디오, 특징 추출, 데이터셋)
├── evaluation/                 # 공통 메트릭 (EER, min-tDCF)
├── configs/                    # 설정 (simulation.yaml)
└── scripts/                    # CLI 진입점
```

## 빠른 시작

```bash
# 1. 환경 설치
conda create -n deepfake python=3.10 -y
conda activate deepfake
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. 전체 시뮬레이션 실행 (공격→방어→분석→보고서)
python -m voice_defense.scripts.run_simulation
```

## 실행 파이프라인

```
┌──────────────────────────────────────────────────────────┐
│                  pipeline/simulate.py                     │
├────────────────────┬─────────────────────────────────────┤
│  PHASE 1: 공격     │  PHASE 2: 방어                      │
│  XTTS v2 복제      │  AASIST / LCNN / RawNet2 탐지       │
│  ECAPA-TDNN 우회   │                                      │
│  ASR 측정          │  PHASE 3: 분석                       │
│                    │  도메인 갭 (FID + t-SNE)             │
├────────────────────┴─────────────────────────────────────┤
│  PHASE 4: 보고서                                          │
│  → outputs/results/simulation_report.md                   │
└──────────────────────────────────────────────────────────┘
```

## 개별 실행

```bash
# 공격만
python -m voice_defense.scripts.run_clone_attack --config configs/simulation.yaml

# 방어만
python -m voice_defense.scripts.run_alt_defense --phase train
python -m voice_defense.scripts.run_alt_defense --phase evaluate_clones

# 공격 Zoo로 학습용 spoof 생성
python -m voice_defense.scripts.run_generate_attacks \
    --prompts-jsonl data/prompts/train.jsonl \
    --out-dir data/spoof_self/train \
    --pipelines synthetic_tts --n-samples 50 --seed 42
```

## 기술 스택

| 영역 | 기술 |
|------|------|
| 음성 복제 | Coqui TTS XTTS v2 |
| 화자 인증 | SpeechBrain ECAPA-TDNN |
| 방어 (1차) | WavLM + AASIST + OC-Softmax |
| 방어 (대안) | LCNN, RawNet2 |
| 공격 Zoo | XTTS, OpenVoice, Tortoise, Bark, RVC, seed-vc |
| 데이터셋 | LibriSpeech, ASVspoof 2019 |

## 평가 프로토콜

| Protocol | 설명 | 목적 |
|----------|------|------|
| `seen` | 학습에 포함된 공격 알고리즘 | sanity check |
| `unseen` | 학습에 안 넣은 Attack Zoo 알고리즘 | 일반화 |
| `wild` | In-the-Wild + 외부 데이터 | 실전 일반화 |
| `redteam` | Red Team 제출 wav | 본 평가 |

## 요구 사항

- Python 3.10
- CUDA GPU (4-6GB VRAM 이상)
- ~10GB 디스크 (데이터셋 + 모델)

## 브랜치

| 브랜치 | 설명 |
|--------|------|
| `main` | 기본 방어 프레임워크 |
| `attack` | **공격-방어 통합 구조 (최신)** — 이 브랜치를 사용하세요 |

## 라이선스

연구·교육 목적. 공격 zoo의 각 모델은 원 라이선스를 따름.
