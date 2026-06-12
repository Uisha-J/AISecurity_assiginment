# 실측 실행 런북 (Milestone D)

코드/배선은 모두 완성되어 **합성 데이터로는 지금 바로 동작**합니다(`tests/test_smoke.py` 16개 통과,
`run_asv_bypass --demo`, `demo_e2e`). 이 문서는 **실제 모델·데이터로 1회 측정**해서 결과를 남기는 절차입니다.
GPU 환경(4–6GB VRAM 이상)과 수 GB 다운로드가 필요해 별도 단계로 둡니다.

## 0. 사전 준비

```bash
conda create -n deepfake python=3.10 -y && conda activate deepfake
# CUDA 빌드 torch (CPU만 있으면 --index-url .../whl/cpu)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .                      # 코어 의존성 + voice_defense 패키지
pip install -e ".[asr]"               # speechbrain (ECAPA-TDNN) — 실측 ASR
pip install TTS                       # Coqui XTTS v2 — 음성 복제 공격
```

## 1. 데이터 준비

```bash
python -m voice_defense.scripts.download_data --list                     # 키 목록
python -m voice_defense.scripts.download_data --datasets librispeech_devclean musan rirs_noises
python -m voice_defense.scripts.download_data --verify                   # 존재 확인
```

- **ASVspoof2019 LA**는 gated(등록 필요)입니다. 받은 뒤 시뮬레이션이 기대하는 위치에 둡니다:
  ```
  <data-root>/asvspoof2019/ASVspoof2019_LA_train/flac/*.flac
  <data-root>/asvspoof2019/ASVspoof2019_LA_dev/flac/*.flac
  <data-root>/asvspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl}.txt
  ```
  (`download_data`는 `voice_defense/data/...`에 받으므로, 시뮬레이션엔 `--data-root`를 맞춰 지정하세요.)

## 2. 방어 모델 학습·평가 (AASIST)

```bash
python -m voice_defense.scripts.train --config configs/default.yaml
python -m voice_defense.scripts.evaluate \
    --checkpoint outputs/checkpoints/best.pt \
    --protocols protocols/seen.yaml protocols/unseen.yaml protocols/wild.yaml \
    --report-dir results/aasist_eval
```

## 3. 전체 Red-vs-Blue 시뮬레이션 (공격→방어→분석→보고서)

```bash
python -m voice_defense.scripts.run_simulation \
    --config configs/simulation.yaml --data-root ./data --output-root ./outputs
```
- 산출물: `outputs/results/simulation_report.md`, `simulation_results.json`, `attack_summary.csv`.
- **B 버그 수정 검증**: alt 탐지기 EER이 정상 범위(<~50%)로 나오는지 확인(과거엔 크래시).

## 4. ASV 우회 실측 (trial→보정→tandem→매핑 리포트)

```bash
# speaker_data.json = {"spk1": ["a.wav","b.wav"], ...}; cloned-dir = 화자별 복제 wav
python -m voice_defense.scripts.run_asv_bypass \
    --speaker-data data/speaker_data.json \
    --cloned-dir outputs/cloned_audio \
    --verifier-ckpt <ecapa_ckpt_or_blank_for_default> \
    --cm-ckpt outputs/models/lcnn_best.pth --cm-name LCNN \
    --output-dir results/asv_bypass_eval --seed 42
```
- 산출물: `protocol.csv`, `calibration.json`, `tandem.json`, `results/asv_mapping_report.md`.

## 5. 결과 커밋

`outputs/`는 gitignore 대상이므로, 보존할 결과는 **tracked 디렉터리**(`results/`)에 복사 후 커밋:

```bash
git add results/ && git commit -m "Add real measurement results (milestone D)"
```

## 정상성 체크리스트 (DoD)

- [ ] EER ∈ [0, 0.5], min-tDCF 정상(NaN 아님)
- [ ] alt 탐지기 EER이 합리적(<~50%) — B1/B2/B3 수정 검증
- [ ] `asr_tandem ≤ asr_asv` (tandem.json 모든 셀)
- [ ] tandem 4사분면 합 = spoof trial 수
- [ ] `simulation_report.md` / `asv_mapping_report.md` 정상 렌더
