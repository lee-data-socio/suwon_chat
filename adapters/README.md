# 어댑터

수원서베이 응답 분포를 예측하도록 학습한 LoRA 어댑터 세 개다. 기본 모델은 `MLP-KTLim/llama-3-Korean-Bllossom-8B`이고,
학습 방법은 SubPOP(UC Berkeley Canny Lab)을 따랐다. LoRA r=8, α=32, 대상 모듈 q_proj·v_proj, 배치 128, 학습률 5e-5.

| 폴더 | 학습 자료 | 정리본에서의 이름 |
|---|---|---|
| `exp1` | 2023·2024년 전체 | `시험1_미래예측/bllossom-8b-b128-lr5e-5_suwon-v2-full_1003857` |
| `exp2` | 7개 집단을 뺀 것 | `시험2_미관측집단/bllossom-8b-b128-lr5e-5_suwon-v2-exp2_1006787` |
| `exp3` | 4개 문항을 뺀 것 | `시험3_새질문/bllossom-8b-b128-lr5e-5_suwon-v2-exp3_1006789` |

`수원서베이_학습결과_정리본/02_모델/`에서 파일을 그대로 복사했다. 실시간 예측(`PREDICT_BACKEND=live`)이 이 폴더를 읽는다.
