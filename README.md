# Voice Deepfake Attack & Defense Simulation

음성 딥페이크 공격과 방어를 **동등한 모듈**로 구성하고,
모의 실험 파이프라인으로 공격 성공률(ASR)과 방어 탐지율을 비교 분석하는 프로젝트.

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
├── common/                     # 공용 유틸리티
│   ├── audio.py                # 오디오 I/O, 리샘플링, 트리밍
│   ├── features.py             # LFCC, Mel-spectrogram 추출
│   ├── redteam_data.py         # LibriSpeech, ASVspoof, ClonedVoice 데이터셋
│   ├── augment.py              # RawBoost, 데이터 증강
│   ├── dataset.py              # 프로토콜 기반 Dataset
│   └── compat.py               # PyTorch 2.7+ / SpeechBrain 호환 패치
│
├── evaluation/                 # 평가 메트릭
│   ├── metrics.py              # EER, min-tDCF, per-attack breakdown
│   ├── eval_aasist.py          # AASIST 전용 평가
│   └── report.py               # 프로토콜별 리포트
│
├── configs/                    # 설정 파일
│   ├── simulation.yaml         # 전체 시뮬레이션 설정
│   └── default.yaml            # AASIST 학습 설정
│
├── protocols/                  # 평가 프로토콜 (seen/unseen/wild)
├── scripts/                    # CLI 진입점
├── tests/                      # smoke / 회귀 테스트
└── data/                       # 데이터 (prompts, seed_speech 등)
```

## 빠른 시작

### 1. 환경 설치

```bash
conda create -n deepfake python=3.10 -y
conda activate deepfake
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. 전체 시뮬레이션 실행

```bash
python -m voice_defense.scripts.run_simulation --config configs/simulation.yaml
```

이 명령 하나로:
1. **공격**: XTTS v2로 음성 복제 → ECAPA-TDNN 화자 인증 우회 → ASR 측정
2. **방어**: LCNN/RawNet2로 딥페이크 탐지
3. **분석**: 도메인 갭 (FID + t-SNE)
4. **보고서**: `outputs/results/simulation_report.md` 생성

### 3. 개별 실행

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

# AASIST 방어 모델 학습
python -m voice_defense.scripts.train --config configs/default.yaml
```

## 실험 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    pipeline/simulate.py                      │
├──────────────────────┬──────────────────────────────────────┤
│                      │                                      │
│  PHASE 1: ATTACK     │  PHASE 2: DEFENSE                   │
│  ┌────────────────┐  │  ┌────────────────┐                 │
│  │ attack/clone/  │  │  │ defense/aasist/ │ AASIST 탐지기  │
│  │ XTTS v2 복제   │  │  ├────────────────┤                 │
│  ├────────────────┤  │  │ defense/alt/    │ LCNN / RawNet2 │
│  │ attack/verify/ │  │  └────────────────┘                 │
│  │ ECAPA-TDNN ASR │  │                                      │
│  └────────────────┘  │  PHASE 3: ANALYSIS                  │
│                      │  ┌─────────────────────┐            │
│                      │  │ defense/domain_gap/  │            │
│                      │  │ FID + t-SNE          │            │
│                      │  └─────────────────────┘            │
├──────────────────────┴──────────────────────────────────────┤
│  PHASE 4: REPORT                                            │
│  pipeline/report.py → outputs/results/simulation_report.md  │
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

## 요구 사항

- Python 3.10
- CUDA GPU (4-6GB VRAM 이상)
- ~10GB 디스크 (데이터셋 + 모델)

## 진행 상황 / 남은 작업

ASV 우회 시나리오(특정인 복제 → 화자인증 통과)를 학술적으로 성립시키기 위한
연결 모듈을 추가하는 중. 설계 근거는 `docs/asv_bypass_design.md` 참고.

### 완료
- [x] `common/trial_protocol.py` — 4분류 trial(enrollment / genuine / impostor / spoof) 생성. 화자 분리 원칙, seed 재현성, CSV·JSON 직렬화.
- [x] `attack/verify/calibrate.py` — genuine vs zero-effort impostor로 ASV의 **EER 운영점 임계값** 산출. 임의 sweep 대신 보정된 임계값에서 ASR 측정.
- [x] `pipeline/tandem.py` — 복제본을 ASV·CM에 동시 통과시켜 4사분면 집계, 핵심 지표 `asr_asv` / `asr_tandem` 산출.
- [x] 위 3개 모듈 스모크 테스트(모델 없이 합성 데이터로 검증) — `tests/test_smoke.py`.

### 남은 작업 (TODO)
- [ ] **공격×방어 매핑 리포트** — `pipeline/report.py` 확장. 행=공격 벡터(xtts, rvc, +codec, +PGD), 열=방어(ASV단독 / LCNN / RawNet2 / AASIST / tandem), 셀=ASR. 가장 취약한 (공격, 방어) 쌍 자동 하이라이트.
- [ ] **CLI 진입점** — `scripts/run_asv_bypass.py`. trial 생성 → ASV 보정 → tandem 평가 → 리포트를 한 줄로 연결.
- [ ] **README ↔ 코드 정합성** — README가 가리키는 `run_clone_attack.py` / `run_alt_defense.py`가 실제 `scripts/`에 없음. 스크립트를 만들거나 README를 실존 스크립트(`run_simulation`, `run_custom_attack`, `run_redteam`)에 맞게 수정.
- [ ] **브랜치 정리** — `main`은 옛 구조. `attack`을 `main`으로 머지하거나 `main`에 안내 명시.
- [ ] **실측 결과 생성** — XTTS(~2GB) / speechbrain ECAPA / WavLM+AASIST / ASVspoof2019 내려받아 1회 실행, 결과 figure·표 커밋. (`requirements.txt`의 주석 처리된 `TTS`, `speechbrain` 의존성 활성화 필요.)

## 라이선스

연구·교육 목적. 공격 zoo의 각 모델은 원 라이선스를 따름.
