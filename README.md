# Voice Defense — 화자 검증 딥페이크 방어 (Blue Team)

공격자가 어떤 방법을 쓸지 모르는 상황에서, 화자 검증(ASV)을 노린 음성 딥페이크를
방어하는 프로젝트. Red/Blue 시뮬레이션의 Blue 측 코드베이스.

## 핵심 전략

> 공격자가 안 쓸 것 같은 것까지 내가 미리 다 써본다.

1. **Attack Zoo** — 가능한 모든 오픈소스 TTS/VC를 내가 먼저 돌려 spoof를 대량 생성
2. **Holdout 규율** — 공격 zoo 중 일부는 학습에 절대 넣지 않고 unseen 평가용으로만 사용
3. **SSL + AASIST + OC-Softmax** — 일반화에 가장 강한 조합
4. **공격적 augmentation** — 코덱, 잔향, 잡음, RawBoost
5. **정직한 평가** — Seen / Unseen / Wild / RedTeam 4개 프로토콜 분리

## 진행 상황

| 단계 | 상태 | 비고 |
|---|---|---|
| Attack Zoo (생성기 + post-process + orchestrator) | **완료** | `synthetic_tts`, `artifact_vc` baseline은 외부 가중치 없이 즉시 동작 |
| Red Team 제출 생성 (`labels.csv`, `manifest.json`) | **완료** | 체크포인트가 있으면 점수 기반 후보 선택까지 |
| 결정성 / manifest / 진행률 / 회귀 테스트 | **완료** | 같은 `--seed`로 byte-identical 재현 |
| 방어 모델 학습 / 평가 (SSL + AASIST + OC-Softmax) | 다음 단계 | `scripts/train.py`, `scripts/evaluate.py` |
| Red Team 제출 평가 | 다음 단계 | `scripts/eval_submission.py` |

## 디렉터리

```
voice_defense/
├── attack_zoo/              # Blue 측이 자체 학습용으로 생성하는 가짜 공격
│   ├── tts/
│   │   ├── synthetic_tts.py # baseline (의존성 X)
│   │   ├── xtts.py / openvoice.py / tortoise.py / bark.py
│   │   └── ...
│   ├── vc/
│   │   ├── artifact_vc.py   # baseline (의존성 X)
│   │   ├── rvc.py / seed_vc.py
│   │   └── ...
│   ├── post_process/        # 코덱, RIR, 노이즈
│   └── orchestrator.py      # 무작위 조합 파이프라인 + manifest 작성
│
├── defense/                 # 방어 모델
│   ├── frontend.py          # WavLM/XLS-R (frozen SSL)
│   ├── backend.py           # AASIST (graph attention)
│   ├── loss.py              # OC-Softmax
│   └── model.py             # End-to-end 통합
│
├── data_pipeline/           # PyTorch Dataset + augmentation
├── training/                # 학습 루프
├── evaluation/              # EER, t-DCF, 리포트
├── redteam/                 # 공격팀 제출 인터페이스 + builder
├── protocols/               # YAML 평가 프로토콜 (seen/unseen/wild)
├── configs/                 # 학습 설정
├── scripts/                 # CLI 진입점 (`python -m voice_defense.scripts.*`)
├── tests/                   # smoke / 회귀 테스트
└── data/
    ├── prompts/             # 공격 생성용 입력 JSONL (예시 포함)
    ├── seed_speech/         # artifact_vc용 짧은 seed wav
    ├── spoof_self/          # 생성된 공격 wav (만든 후 채워짐)
    └── augment/             # RIR / MUSAN 등 (선택)
```

## 빠른 시작

### 1. 환경 설치

```bash
pip install -r requirements.txt
```

baseline 공격 두 개(`synthetic_tts`, `artifact_vc`)만 돌리려면 `numpy + scipy + soundfile + librosa + tqdm`만 있어도 충분합니다. 무거운 의존성(`transformers`, 외부 TTS/VC 라이브러리)은 실제로 그 generator를 사용할 때만 필요.

### 2. 공격 zoo로 학습용 spoof 생성

baseline만 사용 (외부 가중치 불필요):

```bash
python -m voice_defense.scripts.generate_attacks \
    --prompts-jsonl data/prompts/train.jsonl \
    --out-dir data/spoof_self/train \
    --pipelines synthetic_tts \
    --n-samples 50 \
    --seed 42
```

`artifact_vc`도 함께 쓰려면 prompts 줄마다 `target_wav` 경로가 있어야 함:

```bash
python -m voice_defense.scripts.generate_attacks \
    --prompts-jsonl data/prompts/train_with_vc.jsonl \
    --out-dir data/spoof_self/train_vc \
    --pipelines synthetic_tts,artifact_vc \
    --n-samples 50 \
    --seed 42
```

외부 가중치가 깔려 있으면 추가 (미설치 시 자동 skip):

```bash
python -m voice_defense.scripts.generate_attacks \
    --prompts-jsonl data/prompts/train_with_vc.jsonl \
    --out-dir data/spoof_self/train_full \
    --pipelines synthetic_tts,artifact_vc,xtts,rvc,openvoice,tortoise,bark,seed_vc \
    --post-process codec,noise,rir \
    --rir-dir data/augment/rirs \
    --noise-dir data/augment/musan
```

### 3. Red Team 제출 생성

생성한 spoof wav를 다시 변형해 `labels.csv` + `manifest.json` 까지:

```bash
python -m voice_defense.scripts.redteam_attack \
    --input-dir data/spoof_self/train \
    --out-dir redteam/submissions/team_a \
    --variants-per-file 8 \
    --keep-all \
    --seed 42
```

방어 체크포인트가 있으면 `--checkpoint checkpoints/best.pt`를 주어 "방어 점수가 높은 변형만" 선택 가능.

### 4. 방어 모델 학습 / 평가 (다음 단계)

```bash
python -m voice_defense.scripts.train --config configs/default.yaml
python -m voice_defense.scripts.evaluate --checkpoint checkpoints/best.pt
python -m voice_defense.scripts.eval_submission \
    --submission-dir redteam/submissions/team_a \
    --checkpoint checkpoints/best.pt
```

## 입력 JSONL 형식

`data/prompts/*.jsonl` — **한 줄 = JSON object 1개**. (메모장/PowerShell BOM 자동 인식)

| 키 | 누가 쓰나 | 필수 여부 |
|---|---|---|
| `text` | TTS 계열 (`xtts`, `openvoice`, `tortoise`, `bark`) | 해당 generator 사용 시 필수 |
| `target_wav` | VC 계열 (`artifact_vc`, `rvc`, `seed_vc`) | 해당 generator 사용 시 필수 |
| `reference_wav` | 화자 클로닝(`xtts`, `openvoice`, `tortoise`, `seed_vc`) | 선택 (없으면 default) |
| `seed` | 모든 generator | 선택 (없으면 `--seed`에서 자동 파생) |

> `synthetic_tts`는 `text`도 선택 — 없으면 default 텍스트로 동작하므로 "프롬프트 수만큼 다양한 합성 음성"을 빠르게 뽑을 때 좋음.

## 출력물 형식

`generate_attacks` 실행 후 `--out-dir`에는 다음이 생성됩니다.

```
data/spoof_self/train/
├── sample_000000.wav        # 공격 음성
├── sample_000000.json       # 해당 샘플 메타 (algorithm, post_processing, extra)
├── ...
├── sample_NNNNNN.error.txt  # 실패한 샘플의 사유 (있을 때만)
└── manifest.json            # 전체 요약
```

`manifest.json`에는 다음이 들어갑니다.

- `n_requested`, `n_written`, `n_failed` — 성공/실패 카운트
- `pipeline_usage` — 어느 generator(+post_process)가 몇 번 쓰였는지
- `generators_available`, `post_processors_available` — 실제로 활성화된 모듈
- `failures_head` — 첫 10개 실패의 prompt + 에러 메시지
- `created_at` (UTC)

## 재현성

- `--seed` 한 개로 (1) 파이프라인 선택, (2) post-process 선택, (3) 각 generator 내부 난수까지 결정됨.
- prompt에 `seed`가 명시되어 있으면 그것이 우선, 없으면 `--seed`에서 sample index별로 파생.
- 같은 seed + 같은 prompts 파일 → **byte-identical**한 wav 산출 (smoke test로 보장).

## smoke / 회귀 테스트

전체 파이프라인이 외부 데이터/가중치 없이 동작하는지 검증:

```bash
python -m voice_defense.tests.test_smoke
```

포함된 케이스:

- `_load_prompts`의 BOM/missing/empty/잘못된 JSON 처리
- orchestrator 재현성 + manifest 작성
- baseline TTS/VC 생성기 단독 동작
- Red Team 제출 생성기 (`labels.csv`, `manifest.json`, wav 변형 수)
- 데이터 파이프라인, RawBoost, 손실 함수, AASIST backend, 미니 학습 루프

## 평가 프로토콜

| Protocol | 설명 | 목적 |
|---|---|---|
| `seen` | 학습에 포함된 공격 알고리즘으로만 평가 | sanity check |
| `unseen` | Attack Zoo에서 학습에 안 넣은 알고리즘 | 일반화 |
| `wild` | In-the-Wild + MLAAD 등 외부 데이터 | 진짜 일반화 |
| `redteam` | 공격팀이 제출한 wav 파일 | 본 평가 |

## 정직성 원칙

- 학습/평가 분리는 **알고리즘 수준**에서 보장 (같은 TTS 모델로 만든 spoof가 양쪽에 들어가면 안 됨)
- Threshold 튜닝은 **dev set**에서만, eval set 점수 보고 다시 만지면 안 됨
- 모든 메트릭은 신뢰구간 포함 보고

## 라이선스

연구·교육 목적. 공격 zoo의 각 모델은 원 라이선스를 따름.
