# ASV 우회 공격 연결 설계 문서

> 목적: 현재 "딥페이크 음성 생성 → 탐지기(CM) 회피" 수준에 머문 공격을,
> 제안서의 핵심인 **"특정인 목소리 복제 → 화자 인증(ASV) 우회"** 시나리오로
> 학술적으로 성립시키기 위한 누락 모듈 설계.

---

## 1. 배경 — 지금 무엇이 비어 있나

현재 코드에는 두 갈래 공격이 섞여 있다.

| 공격 | 노리는 시스템 | 성공 기준 | 위치 | 상태 |
|------|--------------|-----------|------|------|
| A. CM 회피 | 딥페이크 **탐지기**(CM) | 탐지기가 "진짜"로 오판 | `attack/zoo/`, `attack/redteam_system.py` | 구현됨 |
| B. ASV 우회 | **화자 인증기**(ASV) | 특정인으로 인증 통과 | `attack/clone/` + `attack/verify/` | **절반만** |

제안서가 말하는 공격은 **B**이지만, 현재 B는 다음이 비어 학술적으로 성립하지 않는다.

1. **ASV 임계값 미보정** — 임의 sweep(0.15~0.35)만 함. 실제 운영점(EER threshold)이 아니므로 ASR 수치에 의미 부여 불가.
2. **대조군 부재** — genuine(본인) / zero-effort impostor(타인 원본) / spoof(복제) 세 집단 비교가 없어 "복제가 인증을 뚫었다"를 증명할 수 없음.
3. **Tandem 미연결** — 실제 위협은 "ASV 통과 + CM 회피" 동시 성립인데, ASV와 CM이 한 평가로 묶이지 않음.
4. **공격↔방어 매핑 부재** — 어떤 공격 벡터가 ASV는 뚫고 CM엔 걸리는지 정리된 산출물이 없음.

---

## 2. 목표

리서치·외부모델·데이터 없이, **표준 기법 + 합성/기존 데이터**만으로 구현·검증 가능한
네 개의 연결 모듈을 정의한다. 모델(XTTS/speechbrain/WavLM)과 데이터가 갖춰지면
동일 코드로 실측된다.

---

## 3. 위협 모델 (문제정의)

| 축 | 정의 |
|----|------|
| **목표** | 피해자 화자로 등록된 ASV를, 복제 음성으로 통과(accept)시킴 |
| **지식수준** | black-box(ASV 점수만) / gray-box(임계값 알려짐) / white-box(CM 그래디언트 접근, PGD) |
| **능력** | 피해자 음성 N초 확보(기본 6초). enrollment 음성에는 접근 불가 가정 |
| **방어 배치** | ① ASV 단독 ② CM 단독 ③ **ASV+CM tandem** |

---

## 4. 모듈 설계

### 4.1 ASV 임계값 보정기 — `attack/verify/calibrate.py`

**문제**: 임계값이 임의값이면 ASR이 무의미.
**해법**: genuine/impostor trial로 ASV의 EER 운영점을 먼저 고정한다.

```
입력:  화자별 발화 목록 (LibriSpeech)
처리:
  - genuine 쌍:  같은 화자의 서로 다른 발화 (enroll vs test)
  - impostor 쌍: 다른 화자 간 발화 (zero-effort)
  - 각 쌍의 cosine similarity 분포 → EER 지점 임계값 산출
출력:  {eer, threshold_at_eer, genuine_scores, impostor_scores}
의존:  SpeakerVerifier.similarity(), evaluation.metrics.compute_eer
```

**핵심**: 이렇게 보정한 임계값에서 복제본 ASR을 측정해야
"실제 운영 인증기를 복제가 뚫었다"가 성립.

---

### 4.2 Trial 프로토콜 생성기 — `common/trial_protocol.py`

**문제**: enrollment/평가 집단이 코드에 흩어져 있음.
**해법**: 4분류 trial을 명시적으로 구성.

```
출력 프로토콜 (화자 풀에서 자동 생성):
  - enrollment:        피해자 등록 음성
  - trial_genuine:     본인 다른 발화        (라벨: target,  기대: accept)
  - trial_impostor:    타인 원본 발화         (라벨: nontarget, 기대: reject)
  - trial_spoof:       복제본                 (라벨: spoof,    기대: reject 되어야 정상)
구조: dataclass TrialSet → CSV/JSON 직렬화
원칙: 화자 분리(enroll 화자와 impostor 화자 겹치지 않게)
```

---

### 4.3 Tandem 평가기 — `pipeline/tandem.py`

**문제**: ASV 통과 여부와 CM 탐지 여부가 따로 놂.
**해법**: 복제본을 ASV·CM에 동시 통과시켜 4사분면으로 집계.

```
                  CM: 탐지 실패(회피)     CM: 탐지 성공
ASV: 통과(accept)   ★ 완전 우회 (최악)      ASV는 뚫림, CM이 방어
ASV: 거부(reject)    CM만 작동              둘 다 방어 성공

출력:
  - 4사분면 카운트 + 비율
  - ASR_asv  (ASV 단독 우회율)
  - ASR_tandem (ASV+CM 모두 우회율)  ← 핵심 지표
  - (옵션) min-tDCF: ASV·CM 결합 비용
의존:  SpeakerVerifier + defense.alt.detector.detect_single
```

**스토리텔링 결론**: `ASR_asv`는 높지만 `ASR_tandem`은 낮아야
"CM 추가가 방어에 기여했다"가 수치로 증명됨.

---

### 4.4 공격×방어 매핑 리포트 — `pipeline/report.py` 확장

```
행: 공격 벡터 (xtts, rvc, +codec, +PGD ...)
열: 방어 (ASV단독 / LCNN / RawNet2 / AASIST / tandem)
셀: 우회 성공률(ASR) — 높을수록 그 방어가 그 공격에 취약

자동 도출:
  - "이 공격은 ASV는 뚫지만 LCNN에 걸린다" 같은 문장 생성
  - 가장 취약한 (공격, 방어) 쌍 하이라이트
```

---

## 5. 데이터 흐름 (전체 연결)

```
          [4.2 trial 프로토콜]
                  │
   ┌──────────────┼───────────────┐
   │              │               │
 genuine       impostor         spoof(복제)
   │              │               │
   └──────┬───────┘               │
          ▼                       ▼
   [4.1 ASV 보정]          [attack/clone XTTS]
   EER 임계값 산출                 │
          │                       ▼
          └──────────► [4.3 Tandem 평가] ◄── [defense CM]
                              │
                              ▼
                  [4.4 공격×방어 매핑 리포트]
                              │
                              ▼
                  outputs/results/*.md
```

---

## 6. 평가 지표

| 지표 | 의미 | 어디서 |
|------|------|--------|
| ASV EER | 인증기 자체 오류율 (운영점 보정용) | 4.1 |
| ASR_asv | ASV 단독 우회율 (보정 임계값 기준) | 4.3 |
| ASR_tandem | ASV+CM 동시 우회율 (★핵심) | 4.3 |
| CM EER | 탐지기 오류율 | 기존 |
| FID | 도메인 갭 (학습 spoof vs 복제본) | 기존 `domain_gap` |
| per-attack ASR | 공격 벡터별 취약점 | 4.4 |

---

## 7. 구현 순서 (의존성 순)

1. `common/trial_protocol.py` — 다른 모듈의 입력. 의존 없음.
2. `attack/verify/calibrate.py` — 4.2 + 기존 metrics에만 의존.
3. `pipeline/tandem.py` — 4.1·4.2 + 기존 detector에 의존.
4. `pipeline/report.py` 확장 — 4.3 결과 집계.
5. `scripts/run_asv_bypass.py` — CLI 진입점.
6. 합성 음성 + mock 점수로 단위 검증 (모델 없이).

---

## 8. 지금 못 하는 것 (조사/모델/데이터 필요)

| 항목 | 막힌 이유 |
|------|-----------|
| 실제 XTTS 복제 | TTS 모델 다운로드(~2GB) |
| 실측 ASV 점수 | speechbrain ECAPA-TDNN 모델 |
| AASIST 1차 방어 | WavLM 사전학습 + ASVspoof 데이터 |
| 정확한 인용·SOTA 수치 | 문헌 리서치 (본 단계 제외) |

> 위 4개 모듈은 **로직·구조·합성 데이터로 전부 구현·검증 가능**하며,
> 모델/데이터가 준비되면 코드 수정 없이 그대로 실측된다.

---

## 9. 부록 — 발견된 기존 버그 (점검 중 수정)

| 파일 | 버그 | 수정 |
|------|------|------|
| `common/audio.py` | 신버전 torchaudio가 `load`를 torchcodec로 위임해 실패 | soundfile+librosa 기반으로 교체 |
| `defense/alt/models.py` (LCNN) | FC 크기를 더미 200프레임으로 고정 → 입력 길이 다르면 shape 에러 | AdaptiveAvgPool2d 추가로 길이 무관화 |
