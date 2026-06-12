# Cross-Attack Generalization Matrix

- 화자 수: 20, 공격당 클론: {'xtts': 60, 'synthetic_tts': 60, 'artifact_vc': 60}
- 값 = 탐지율(%). 행 = 학습 공격, 열 = 테스트 공격. 대각선 = seen.

| train＼test | xtts | synthetic_tts | artifact_vc |
|---|---|---|---|
| **xtts** | 100.0% (seen) | 0.0% | 71.7% |
| **synthetic_tts** | 33.3% | 100.0% (seen) | 100.0% |
| **artifact_vc** | 0.0% | 51.7% | 100.0% (seen) |

## 해석

- 다른 공격으로 학습한 CM의 **XTTS 탐지율 평균 16.7%** (범위 0~33%) → 현대 신경망 복제의 미학습 탐지 한계.