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

## 디렉터리

```
voice_defense/
├── attack_zoo/         # Blue 측이 자체 학습용으로 생성하는 가짜 공격
│   ├── tts/            # XTTS, OpenVoice, Tortoise, Bark 어댑터
│   ├── vc/             # RVC, seed-vc 어댑터
│   ├── post_process/   # 코덱, RIR, 노이즈
│   └── orchestrator.py # 무작위 조합 파이프라인
│
├── defense/            # 방어 모델
│   ├── frontend.py     # WavLM/XLS-R (frozen SSL)
│   ├── backend.py      # AASIST (graph attention)
│   ├── loss.py         # OC-Softmax
│   └── model.py        # End-to-end 통합
│
├── data_pipeline/      # PyTorch Dataset + augmentation
├── training/           # 학습 루프
├── evaluation/         # EER, t-DCF, 리포트
├── redteam/            # 공격팀 제출 인터페이스
├── protocols/          # YAML 평가 프로토콜 (seen/unseen/wild)
├── configs/            # 학습 설정
└── scripts/            # CLI 진입점
```

## 빠른 시작

```bash
# 1. 환경
pip install -r requirements.txt

# 2. 공격 zoo로 학습용 spoof 생성
python -m scripts.generate_attacks \
    --bonafide-dir data/bonafide_train \
    --output-dir data/spoof_self/train \
    --pipelines xtts,rvc,openvoice \
    --n-samples 5000

# 3. 방어 모델 학습
python -m scripts.train --config configs/default.yaml

# 4. 평가 (4개 프로토콜 모두)
python -m scripts.evaluate --checkpoint checkpoints/best.pt

# 5. Red Team 제출 평가
python -m scripts.eval_submission \
    --submission-dir redteam/submissions/team_a \
    --checkpoint checkpoints/best.pt
```

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
