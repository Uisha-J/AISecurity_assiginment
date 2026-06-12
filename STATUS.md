# 프로젝트 현황 (STATUS)

> 이 문서 하나면 **지금 뭐가 됐고, 어디까지 왔고, 다음에 뭘 해야 하는지** 바로 파악됩니다.
> 처음 보는 사람은 이 파일 → [README.md](README.md) → [docs/runbook_real_measurement.md](docs/runbook_real_measurement.md) 순서로 보세요.

## 0. 이 프로젝트가 뭔가요

음성 딥페이크 **공격(Red Team)과 방어(Blue Team)를 동등하게 구현**하고, 명령 한 줄로
공격 → 방어 → 분석 → **보고서**까지 자동으로 돌려 비교하는 모의실험 프레임워크입니다.

- 공격: XTTS v2로 목소리 복제 → 화자 인증(ECAPA-TDNN) 우회. 지표 = **ASR(공격 성공률)**
- 방어: LCNN / RawNet2 / AASIST로 가짜 음성 탐지. 지표 = **EER(탐지 오류율)**
- ASV 우회 연결: 복제본을 ASV와 탐지기(CM)에 **동시 통과**시키는 tandem 평가 + 공격×방어 매트릭스

---

## 1. 지금까지 된 것 ✅ (코드/문서/테스트 = 완료)

- [x] **공격 zoo**(XTTS/RVC/합성 베이스라인 + 코덱·RIR·노이즈 후처리) + 화이트박스 PGD 레드팀
- [x] **방어 모델**: AASIST(WavLM+그래프어텐션+OC-Softmax), LCNN/RawNet2, 도메인갭(FID/t-SNE)
- [x] **ASV 우회 통합**: `common/trial_protocol.py`, `attack/verify/calibrate.py`, `pipeline/tandem.py`
- [x] **공격×방어 매핑 리포트** `pipeline/report.py::generate_asv_mapping_report()`
- [x] **CLI** `scripts/run_asv_bypass.py` — 모델 없이 `--demo`로도 4개 산출물 생성
- [x] **치명 버그 수정**: LCNN/RawNet2 탐지기의 `compute_eer` 인자순서·튜플 버그(학습이 크래시하던 것), 라벨 규약(1=bonafide) 통일
- [x] **패키징/위생**: `pyproject.toml`(`pip install -e .`), GitHub CI, LICENSE/THIRD_PARTY, `.gitignore`
- [x] **실측 턴키 준비**: `scripts/setup_data.py`(데이터 자동 배치), `scripts/preflight.py`(환경 점검), 데이터 경로 불일치 수정
- [x] **스모크 테스트 16개 통과** (모델·데이터·GPU 없이)

검증 명령: `python -m pytest tests/test_smoke.py -q` → `16 passed`

---

## 2. 지금 어디까지 왔나 / 남은 것 🟡

- **코드·문서·테스트·통합은 전부 완료**되어 `main` 브랜치에 있습니다.
- **남은 단 하나 = 실제 GPU에서 실측 1회 실행** (보고서용 진짜 숫자 생성).
  - 이건 GPU + 데이터셋이 필요해 개발 환경에서 미리 못 돌렸습니다. 코드/배선은 검증됐지만
    실제 GPU 첫 실행은 직접 해봐야 합니다.

---

## 3. 이제 뭘 해야 하나 👉 (다음 사람 액션)

### STEP ★ 지금 바로 — ASVspoof2019 데이터셋 신청 (병목)
- https://datashare.ed.ac.uk/handle/10283/3336 에서 **가입 후 LA.zip 다운로드**
- **승인이 최대 ~24시간** 걸리므로 **가장 먼저** 신청해두고 나머지를 준비하세요.
- (LibriSpeech·모델 가중치는 자동 다운로드라 가입 불필요. 가입·수동이 필요한 건 ASVspoof 하나뿐.)

### 그다음 — 원격 GPU에서 실행
전체 절차는 **[docs/runbook_real_measurement.md](docs/runbook_real_measurement.md)** 에 단계별로 있습니다. 요약:
```bash
pip install -e . && pip install -e ".[asr]" && pip install TTS   # 환경
python -m voice_defense.scripts.setup_data --data-root ./data     # LibriSpeech 자동 배치
#  ASVspoof2019 LA를 ./data/asvspoof2019/ 에 배치 (런북 STEP 1b)
python -m voice_defense.scripts.preflight --data-root ./data      # 실행 전 점검 (FAIL 없으면 OK)
python -m voice_defense.scripts.run_simulation --config configs/simulation.yaml --data-root ./data --output-root ./outputs
#  → outputs/results/simulation_report.md (보고서)
python -m voice_defense.scripts.run_asv_bypass --speaker-data data/speaker_data.json \
    --cloned-dir outputs/cloned_audio --verifier-ckpt '' --cm-ckpt outputs/models/lcnn_best.pth \
    --output-dir results/asv_bypass_eval --device cuda           # tandem 매트릭스 (런북 STEP 4)
```
- ⏱ 셋업 ~1시간 + 연산 1.5~6시간. 마감이 급하면 런북의 "마감 단축"(화자 20→5, 에폭 30→8)으로 ~1시간.

---

## 4. 모델/데이터 없이 "어떻게 도는지"만 바로 보고 싶다면 (다운로드 0)
```bash
pip install -e .
python -m pytest tests/test_smoke.py -q                 # 16 passed
python -m voice_defense.scripts.run_asv_bypass --demo   # 공격×방어 매트릭스 리포트 생성(합성)
python -m voice_defense.scripts.demo_e2e                # 공격→방어→평가→리포트 전체 배선(합성)
```

---

## 5. 핵심 파일 안내

| 파일 | 용도 |
|---|---|
| [README.md](README.md) | 프로젝트 전체 개요 · 구조 · 빠른 시작 |
| [docs/runbook_real_measurement.md](docs/runbook_real_measurement.md) | **실측(풀 벤치마크) 단계별 런북** |
| `scripts/setup_data.py` | 데이터셋을 코드가 읽는 정확한 위치로 배치 |
| `scripts/preflight.py` | 실행 전 환경·데이터 1초 점검(OK/WARN/FAIL) |
| `scripts/run_simulation.py` | 공격→방어→분석→보고서 전체 파이프라인 |
| `scripts/run_asv_bypass.py` | ASV 우회 tandem(공격×방어) 평가 |
| `tests/test_smoke.py` | 회귀 테스트 16개(모델 없이) |

---

## 6. 결과물(보고서)이 나오는 위치
- `outputs/results/simulation_report.md` — 공격 ASR + 방어 EER + 도메인갭 FID 종합 보고서
- `results/asv_bypass_eval/results/asv_mapping_report.md` — 공격×방어 ASR 매트릭스(★ 최취약 셀)
- `outputs/results/*.csv`, `*.json` — 원시 수치
- `outputs/cloned_audio/<화자>/*.wav` — 복제 음성 증거

> 참고: Windows 로컬에서 torch import 시 에러가 나면 `KMP_DUPLICATE_LIB_OK=TRUE` 환경변수를 설정하세요.
> AASIST EER 행은 선택(고급)입니다 — 런북 부록 참고. LCNN이 이미 방어 EER을 제공합니다.
