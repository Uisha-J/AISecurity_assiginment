# Voice Deepfake Attack & Defense Simulation

음성 딥페이크 **공격과 방어를 동등한 모듈**로 구현하고,
모의 실험 파이프라인으로 공격 성공률(ASR)과 방어 탐지율을 자동 비교·분석하는 프로젝트.

> 명령어 한 줄이면 공격 → 방어 → 분석 → 보고서까지 자동으로 돌아갑니다.

## 무엇을 하는 프로젝트인가요?

| | Red Team (공격) | Blue Team (방어) |
|---|---|---|
| **목표** | AI로 목소리를 복제해 화자 인증(ASV) 돌파 | 가짜 음성을 탐지해 차단 |
| **핵심 기술** | XTTS v2 음성 복제 + ECAPA-TDNN 우회 | AASIST + LCNN + RawNet2 탐지기 |
| **측정 지표** | ASR (공격 성공률) | EER (탐지 오류율) |

공격과 방어를 동등하게 만들어 자동으로 붙여보고 결과를 리포트로 뽑아주는 것이 핵심.

## 구조

```
├── attack/                     # Red Team — 공격
│   ├── clone/                  # XTTS v2 음성 복제
│   ├── verify/                 # ECAPA-TDNN 화자 인증 우회 + ASR 측정
│   ├── zoo/                    # 다중 공격 생성기 (XTTS, OpenVoice, Tortoise, Bark, RVC 등)
│   │   ├── tts/                # TTS 기반 공격 어댑터
│   │   ├── vc/                 # Voice Conversion 공격 어댑터
│   │   ├── post_process/       # 코덱, RIR, 노이즈 후처리
│   │   └── orchestrator.py     # 무작위 파이프라인 조합 + manifest
│   ├── redteam_system.py       # Red Team 제출 생성 (PGD white-box 포함)
│   └── evaluate_submission.py  # 제출 평가
│
├── defense/                    # Blue Team — 방어
│   ├── aasist/                 # SSL + AASIST + OC-Softmax (1차 방어)
│   │   ├── frontend.py         # WavLM / XLS-R (frozen SSL)
│   │   ├── backend.py          # AASIST graph-attention network
│   │   ├── loss.py             # OC-Softmax loss
│   │   ├── model.py            # End-to-end SpoofDetector
│   │   └── train.py            # 학습 루프
│   ├── alt/                    # LCNN + RawNet2 (대안 탐지기)
│   │   ├── models.py           # LCNN (LFCC 기반), RawNet2 (SincNet 기반)
│   │   ├── detector.py         # 추론 + 클론 음성 평가
│   │   └── train.py            # 학습 + 파인튜닝
│   └── domain_gap/             # 도메인 갭 분석
│       └── analysis.py         # FID + t-SNE (ASVspoof vs XTTS vs Real)
│
├── pipeline/                   # 모의 실험 오케스트레이터
│   ├── simulate.py             # 공격 → 방어 → 분석 → 보고서
│   └── report.py               # Markdown 보고서 생성
│
├── common/                     # 공용 유틸 (audio, features, dataset, augment, compat)
├── evaluation/                 # 평가 메트릭 (EER, min-tDCF, per-attack breakdown)
├── configs/                    # 설정 (simulation.yaml, default.yaml)
├── docs/                       # 설계 문서 (asv_bypass_design.md 등)
├── protocols/                  # 평가 프로토콜 (seen/unseen/wild)
├── scripts/                    # CLI 진입점
├── tests/                      # smoke / 회귀 테스트
└── data/                       # 데이터 (prompts, seed_speech 등)
```

## 빠른 시작

```bash
# 1. 환경 설치
conda create -n deepfake python=3.10 -y
conda activate deepfake
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. 전체 시뮬레이션 실행 (공격→방어→분석→보고서)
python -m voice_defense.scripts.run_simulation --config configs/simulation.yaml
```

위 명령 하나로:
1. **공격**: XTTS v2로 음성 복제 → ECAPA-TDNN 화자 인증 우회 → ASR 측정
2. **방어**: LCNN/RawNet2로 딥페이크 탐지
3. **분석**: 도메인 갭 (FID + t-SNE)
4. **보고서**: `outputs/results/simulation_report.md` 생성

### 개별 실행

```bash
# 공격만 (LibriSpeech 화자 복제)
python -m voice_defense.scripts.run_clone_attack --config configs/simulation.yaml

# 커스텀 음성 공격 (팀원 목소리 등 직접 녹음 파일)
python -m voice_defense.scripts.run_custom_attack --voice-dir data/custom_voices

# 방어 (LCNN/RawNet2 학습·평가)
python -m voice_defense.scripts.run_alt_defense --phase train
python -m voice_defense.scripts.run_alt_defense --phase evaluate_clones

# 공격 Zoo로 학습용 spoof 생성
python -m voice_defense.scripts.run_generate_attacks \
    --prompts-jsonl data/prompts/train.jsonl \
    --out-dir data/spoof_self/train --pipelines synthetic_tts --n-samples 50 --seed 42

# AASIST 방어 모델 학습
python -m voice_defense.scripts.train --config configs/default.yaml
```

## 실험 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    pipeline/simulate.py                      │
├──────────────────────┬──────────────────────────────────────┤
│  PHASE 1: ATTACK     │  PHASE 2: DEFENSE                     │
│  ┌────────────────┐  │  ┌────────────────┐                   │
│  │ attack/clone/  │  │  │ defense/aasist/ │ AASIST 탐지기    │
│  │ XTTS v2 복제   │  │  ├────────────────┤                   │
│  ├────────────────┤  │  │ defense/alt/    │ LCNN / RawNet2   │
│  │ attack/verify/ │  │  └────────────────┘                   │
│  │ ECAPA-TDNN ASR │  │                                       │
│  └────────────────┘  │  PHASE 3: ANALYSIS                    │
│                      │  ┌─────────────────────┐              │
│                      │  │ defense/domain_gap/  │ FID + t-SNE  │
│                      │  └─────────────────────┘              │
├──────────────────────┴──────────────────────────────────────┤
│  PHASE 4: REPORT                                             │
│  pipeline/report.py → outputs/results/simulation_report.md   │
└─────────────────────────────────────────────────────────────┘
```

## 핵심 메트릭

| Metric | 설명 | 누가 쓰나 |
|--------|------|-----------|
| **ASR** | Attack Success Rate — 복제 음성 인증 우회 성공률 | 공격 |
| **EER** | Equal Error Rate — FAR=FRR 교차점 | 방어 |
| **min-tDCF** | minimum tandem Detection Cost Function | 방어 |
| **FID** | Frechet Inception Distance — 스펙트로그램 분포 거리 | 분석 |

## 기술 스택

| 영역 | 기술 |
|------|------|
| 음성 복제 | Coqui TTS XTTS v2 |
| 화자 검증 | SpeechBrain ECAPA-TDNN |
| 방어 (1차) | WavLM + AASIST + OC-Softmax |
| 방어 (대안) | LCNN (LFCC), RawNet2 (SincNet) |
| 공격 Zoo | XTTS, OpenVoice, Tortoise, Bark, RVC, seed-vc |
| 데이터셋 | LibriSpeech, ASVspoof 2019/2021 |
| 프레임워크 | PyTorch, torchaudio |

## 평가 프로토콜

| Protocol | 설명 | 목적 |
|----------|------|------|
| `seen` | 학습에 포함된 공격 알고리즘 | sanity check |
| `unseen` | 학습에 안 넣은 Attack Zoo 알고리즘 | 일반화 |
| `wild` | In-the-Wild + 외부 데이터 | 실전 일반화 |
| `redteam` | Red Team 제출 wav | 본 평가 |

## 설계 문서

- [docs/asv_bypass_design.md](docs/asv_bypass_design.md) — ASV 우회 공격 연결 설계 (trial 프로토콜 / ASV 임계값 보정 / tandem 평가 / 공격×방어 매핑)

## 요구 사항

- Python 3.10
- CUDA GPU (4-6GB VRAM 이상)
- ~10GB 디스크 (데이터셋 + 모델)

## 브랜치

| 브랜치 | 설명 |
|--------|------|
| `main` | 통합 최신본 |
| `attack` | 공격-방어 통합 작업 브랜치 |

## 라이선스

연구·교육 목적. 공격 zoo의 각 모델은 원 라이선스를 따름.
