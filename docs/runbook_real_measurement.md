# 실측 실행 런북 — 풀 벤치마크 (Tier A)

원격 GPU 머신(VSCode Remote, CUDA)에서 **클론 → 실측 → 보고서**까지 그대로 따라 하는 절차입니다.
모든 경로·플래그·데이터셋 키는 실제 코드(`pipeline/simulate.py`, `common/redteam_data.py`,
`scripts/*.py`, `configs/*.yaml`)에 대조해 검증했습니다. **전부 repo 루트에서, bash 셸**로 실행하세요.

> ⏱ 단일 V100/A100급 GPU(여유 VRAM ≥8GB) 기준 **셋업 ~1시간 + 연산 ~1.5–6시간**(LCNN 학습이 대부분).
> ⚠️ **STEP 1b(ASVspoof 가입 다운로드)를 가장 먼저 시작**하세요 — 승인이 최대 ~24시간 걸려서 병목입니다.

핵심 주의: 데이터는 **하나의 표준 루트 `./data`** 로 모읍니다. `scripts/setup_data.py`가 LibriSpeech를
코드가 기대하는 정확한 위치에 배치해줍니다(이걸 안 쓰면 경로 불일치로 공격 단계가 빈손이 됩니다).

---

## STEP 0 — 클론 + 환경 (~15분, ~6GB)

```bash
git clone <repo-url> AISecurity_deepfake && cd AISecurity_deepfake
conda create -n deepfake python=3.10 -y && conda activate deepfake

# CUDA 11.8 PyTorch (박스의 CUDA에 맞게; 최신이면 cu121)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .            # 코어 voice_defense 패키지
pip install -e ".[asr]"     # speechbrain (ECAPA-TDNN, 실측 ASR)
pip install TTS             # Coqui XTTS v2 (음성 복제)

# 0a. GPU 인식 확인 (False면 진행 금지)
python -c "import torch; print('CUDA', torch.cuda.is_available())"

# 0b. (권장) 무거운 모델 미리 캐싱 — 첫 실행이 진행바 없이 멈춘 듯 보이는 것 방지
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2').to('cuda')"   # ~2GB
python -c "from voice_defense.attack.verify.speaker_verifier import SpeakerVerifier; SpeakerVerifier()" # ~300MB
```

---

## STEP 1 — 데이터 (표준 루트 `./data`로 배치)

```bash
# 1a. LibriSpeech (공격 대상 화자) 자동 다운로드 + 정확한 위치 배치
python -m voice_defense.scripts.setup_data --data-root ./data --subset test-clean
#   -> ./data/librispeech/LibriSpeech/test-clean/ 생성 (~378MB)
#   디스크/시간 아끼려면 --subset dev-clean (337MB) 도 동일하게 작동
```

```bash
# 1b. ASVspoof2019 LA — GATED(가입 필요, ~25GB). 방어(LCNN)·도메인갭에 필수. 가장 먼저 신청!
#   1) 가입+다운로드: https://datashare.ed.ac.uk/handle/10283/3336  (LA.zip)
#   2) 아래 트리가 되도록 배치 (setup_data가 위치를 안내해 줌):
unzip LA.zip -d /tmp/la && mkdir -p ./data/asvspoof2019 && cp -r /tmp/la/LA/* ./data/asvspoof2019/
#   최종 형태(반드시 이 경로여야 함):
#     ./data/asvspoof2019/ASVspoof2019_LA_train/flac/*.flac
#     ./data/asvspoof2019/ASVspoof2019_LA_dev/flac/*.flac
#     ./data/asvspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl}.txt
```

---

## STEP 2 — 긴 실행 전에 레이아웃 검증 (<1분, 모든 "조용한 스킵" 사전 차단)

```bash
python -m voice_defense.scripts.setup_data --data-root ./data --skip-librispeech
#  요약에서 LibriSpeech=OK, ASVspoof2019=OK 두 줄이 나와야 함. MISSING이면 STEP 1로.
```

---

## STEP 3 — 풀 시뮬레이션 (공격 ASR + LCNN 방어 EER + 도메인갭 FID + 리포트)

```bash
python -m voice_defense.scripts.run_simulation \
    --config configs/simulation.yaml \
    --data-root ./data \
    --output-root ./outputs 2>&1 | tee simulation_run.log
```
- 로그에서 **`Training alternative detector...`** 가 보이면 LCNN 학습이 도는 것. 대신
  **`ASVspoof2019 not found ... skipping`** 가 보이면 멈추고 STEP 1b 경로를 고치세요.
- ⏱ 약 3.5–6시간(LCNN 30에폭 학습이 대부분).

**마감 단축**(먼저 `configs/simulation.yaml` 편집):
- `attack.num_target_speakers: 20 → 5` (클론 180→45개, 공격 단계 ~6분)
- `defense.alt_detector.epochs: 30 → 8` (LCNN ~30–45분, EER 약간 상승 — 보고서에 명시)
- VRAM <6GB면 `device: cpu` (XTTS가 ~10배 느려짐)

**산출물:**
```
outputs/results/simulation_report.md      <- 메인 서술형 보고서
outputs/results/simulation_results.json   <- attack.asr / defense_alt.eer / domain_gap.fid_*
outputs/results/attack_summary.csv        <- ASR (참조길이 × 임계값)
outputs/results/alt_detector_eval.csv     <- LCNN 파일별 점수
outputs/models/lcnn_best.pth              <- 학습된 탐지기(STEP 4에서 사용)
outputs/cloned_audio/<spk>/{ref_,clone_}*.wav
```

---

## STEP 4 — ASV 우회 tandem (공격×방어 ASR 매트릭스) (~5–15분)

```bash
# 4a. 복제 참조에서 speaker_data.json 생성 (화자당 ref ≥2개 보장)
python - <<'PY'
import json, glob
from pathlib import Path
sd = {}
for d in sorted(Path("outputs/cloned_audio").iterdir()):
    if d.is_dir():
        refs = sorted(glob.glob(str(d / "ref_*.wav")))
        if len(refs) >= 2:
            sd[d.name] = refs
Path("data").mkdir(exist_ok=True)
json.dump(sd, open("data/speaker_data.json", "w"), indent=2)
print(f"wrote {len(sd)} speakers -> data/speaker_data.json")
PY

# 4b. 실모델 tandem 평가 (verifier-ckpt 공백 => 기본 ECAPA 자동 로드, cm-ckpt = STEP 3의 LCNN)
python -m voice_defense.scripts.run_asv_bypass \
    --speaker-data data/speaker_data.json \
    --cloned-dir outputs/cloned_audio \
    --verifier-ckpt '' \
    --cm-ckpt outputs/models/lcnn_best.pth --cm-name LCNN --attack-name XTTS \
    --cm-threshold 0.5 \
    --output-dir results/asv_bypass_eval --device cuda --seed 42

# 4c. 불변식 확인(반드시 성립)
python -c "import json; d=json.load(open('results/asv_bypass_eval/tandem.json')); \
print('asr_asv=%.4f asr_tandem=%.4f'%(d['asr_asv'],d['asr_tandem'])); \
assert d['asr_tandem']<=d['asr_asv']+1e-6; print('OK: tandem<=asv')"
```
산출물: `results/asv_bypass_eval/{protocol.csv, calibration.json, tandem.json, results/asv_mapping_report.md}`

---

## STEP 5 — 보고서용 결과 수집

```bash
cat outputs/results/simulation_report.md
cat results/asv_bypass_eval/results/asv_mapping_report.md
python -c "import json; print(json.dumps(json.load(open('outputs/results/simulation_results.json')), indent=2, ensure_ascii=False))"
# 보존: outputs/는 gitignore 대상 -> 보고서에 넣을 결과는 results/(tracked)로 복사
cp outputs/results/simulation_report.md results/ ; cp outputs/results/*.csv results/
```

| 보고서 섹션 | 파일 | 생성 단계 |
|---|---|---|
| 공격 ASR (XTTS vs ECAPA, 참조길이×임계값) | `attack_summary.csv`, `simulation_results.json["attack"]` | STEP 3 |
| 방어 EER (LCNN) + 정확도/탐지율/FRR | `alt_detector_eval.csv`, `...["defense_alt"]` | STEP 3 |
| 도메인갭 FID (ASVspoof vs 클론 vs 실제) | `simulation_results.json["domain_gap"]` | STEP 3 |
| 메인 서술 보고서 | `outputs/results/simulation_report.md` | STEP 3 |
| 공격×방어 tandem 매트릭스 (★ 최취약 셀) | `results/asv_bypass_eval/results/asv_mapping_report.md` | STEP 4 |
| 음성 증거 (참조 + 클론) | `outputs/cloned_audio/<spk>/*.wav` | STEP 3 |

---

## (부록) AASIST EER 행을 추가하고 싶다면 — 고급/선택

`run_simulation`의 **LCNN이 이미 방어 EER을 제공**하므로 보고서에는 충분합니다. AASIST(WavLM+그래프어텐션)
행까지 원하면 추가로:
- WavLM(~350MB 자동 다운) + GPU VRAM ~8–12GB + 학습 수 시간 필요.
- **주의(중요):** `scripts/train`은 `ProtocolDataset`(YAML 프로토콜)으로 학습하는데, ASVspoof의
  `ASVspoof2019_LA_train/flac/`에는 **bonafide와 spoof가 한 폴더에 섞여** 있고 라벨은 `.trn.txt`에
  있습니다. 따라서 그 폴더를 통째로 한쪽으로 glob하면 **오라벨**됩니다. AASIST를 제대로 하려면
  (a) `.trn` 프로토콜대로 파일을 bonafide/ spoof/ 폴더로 분리한 뒤 `protocols/*.yaml`을 만들거나,
  (b) bonafide=LibriSpeech 실음성, spoof=`outputs/cloned_audio`의 XTTS 클론으로 별도 프로토콜을 구성하되
  화자를 train/dev로 분리(누수 방지)해야 합니다.
- 그래서 본 런북은 AASIST를 **선택**으로 두고 라벨이 올바른 **LCNN**을 메인 방어 지표로 사용합니다.

---

## 실패 빠른 진단
- 공격 단계가 클론 0개 → `./data/librispeech/LibriSpeech/test-clean/` 비었거나 TTS 미설치 (STEP 1a).
- 로그에 `skipping alt-detector` → ASVspoof 경로 불일치 (STEP 1b/2). `outputs/models/lcnn_best.pth`도 안 생김.
- `simulation_results.json`에 `attack/defense_alt/domain_gap` 키가 다 있으면 정상.
- STEP 4c 불변식(`asr_tandem ≤ asr_asv`) 통과해야 함.
- 학습이 멈춘 듯하면 DataLoader `num_workers=4`가 원인일 수 있음 → 0–2로 낮춰 재시도.
