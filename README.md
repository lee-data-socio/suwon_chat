# 수원서베이 시연 챗봇

수원서베이 응답 분포 예측 실험의 결과를 채팅과 차트로 보여 주는 시연 화면이다.
무엇을 왜 만드는지는 [docs/PRD.md](docs/PRD.md)에 있다.

## 준비

저장소만 받으면 된다. 앱이 읽는 원본 자료(`source_data/`, 44MB)와 어댑터(`adapters/`, 40MB)가 저장소에 들어 있다.

```bash
conda create -n llm python=3.12 && conda activate llm
pip install -r requirements.txt
cp .env.example .env               # ANTHROPIC_API_KEY를 채운다. 없으면 추천 질문만 저장된 답변으로 나온다
```

- `source_data/`는 `수원서베이_학습결과_정리본`에서 앱과 테스트에 필요한 파일만 고른 것이다(`scripts/make_source_data.py`). 정리본 폴더 전체가 이 폴더 바로 아래에 있으면 그쪽을 읽는다.
- 윈도우에서는 `scripts/serve.sh` 대신 아래 "실행"의 `python -m …` 명령을 쓴다.

## 실행

```bash
cd backend
python -m suwon_chat.build_index   # 정리본 → data/suwon.db. 끝에 자체 점검을 돈다
python -m suwon_chat.reports       # reports/*.md → 참고 문서 색인. 처음 한 번은 임베딩 모델(약 470MB)을 내려받는다
python -m uvicorn suwon_chat.api:app --port 8000
```

`scripts/serve.sh`는 이 둘을 한 번에 한다.

화면은 한 번 빌드해 두면 같은 서버가 내보낸다. http://localhost:8000 을 연다.

```bash
cd frontend
npm install
npm run build                      # frontend/dist 가 생긴다. 서버를 다시 켤 필요는 없다
```

화면을 고치면서 볼 때는 `npm run dev`를 쓴다 (http://localhost:5173, `/api` 요청은 8000번 서버로 넘어간다).

주소로 장면을 바로 열 수 있다. 예: `http://localhost:8000/?q=q132_1_W2025&a=TENURE&m=exp1&with=last_year`
(`q` 문항, `a` 속성, `m` 어댑터, `with` 함께 볼 것: `last_year` · `reference` · `zero_shot`)

## 테스트

```bash
cd backend
python -m pytest -q tests
```

칸 단위 오차를 정리본 결과표(`01_결과표/*/칸단위.csv`)의 값과 견준다. 대화 테스트는 가짜 클라이언트를 쓰므로 API 키가 없어도 된다.

실제 대화 모델의 답변은 따로 확인한다. 서버를 띄운 채로:

```bash
python scripts/check_answers.py 3  # 추천 질문을 3번씩 돌려 답변 속 숫자가 도구 결과에 있는 값인지 본다 (요금이 든다)
```

## 실시간 버전 (Bllossom을 직접 돌린다)

기본은 **조회 버전**이다. 연구실 GPU 서버에서 미리 계산해 정리본에 저장해 둔 예측을 `data/suwon.db`에서 꺼내 보여 준다. GPU가 필요 없다.

**실시간 버전**은 기본 모델(`MLP-KTLim/llama-3-Korean-Bllossom-8B`)을 이 PC에 올리고, `adapters/`의 LoRA 어댑터 세 개를 갈아 끼우며 질문마다 직접 추론한다. 화면의 "사전 계산" 표시가 "실시간"으로 바뀌고, 목록에 없는 새 문항도 물어볼 수 있다(실측이 없으므로 "실험적"으로 표시된다).

### 필요한 장비

| 장비 | 올리는 방법 | 비고 |
|---|---|---|
| NVIDIA GPU, 메모리 16GB 이상 | 그대로 | 저장된 예측과 거의 같은 값이 나온다. 질문 하나에 몇 초 |
| NVIDIA GPU, 메모리 8~12GB | `LIVE_4BIT=1` (`pip install bitsandbytes`) | 4비트로 줄여 올린다. 값이 조금 달라진다 |
| Apple Silicon, 메모리 32GB 이상 | 그대로 (`mps`) | 돌려 보지 못했다. bfloat16에서 오류가 나면 `LIVE_DTYPE=float16` (값이 조금 달라진다) |
| Intel GPU (`xpu`) | 그대로 | 정리본의 `predict_local.py`가 쓴 방식. 돌려 보지 못했다 |
| GPU 없음, RAM 32GB 이상 | 그대로 (`cpu`) | 된다고 해도 질문 하나에 수십 초~몇 분 |
| RAM 16GB 이하이고 GPU도 없음 | 안 된다 | 조회 버전을 쓴다 |

모델 가중치는 약 16GB이고 처음 한 번 허깅페이스에서 내려받는다(`~/.cache/huggingface`).

### 순서

```bash
# 1. 저장소를 받고 패키지를 깐다 (README 맨 위의 "준비"와 같다)
conda create -n llm python=3.12 && conda activate llm
pip install -r requirements.txt          # NVIDIA GPU가 있으면 먼저 그 PC의 CUDA에 맞는 torch를 깐다 (pytorch.org)

# 2. 색인을 만든다
cd backend
python -m suwon_chat.build_index
python -m suwon_chat.reports

# 3. 실시간 예측이 저장된 예측과 같은 값을 내는지 확인한다 (모델을 처음 내려받는다)
python -m suwon_chat.live_check          # 빠른 점검: 어댑터마다 문항 × 속성 24쌍
python -m suwon_chat.live_check --all    # 전부(어댑터마다 3,729칸). 시연에 쓰기 전에 한 번은 돌린다

# 4. 서버를 실시간 모드로 띄운다. .env 에 PREDICT_BACKEND=live 를 적어도 된다
PREDICT_BACKEND=live python -m uvicorn suwon_chat.api:app --port 8000
#   윈도우 PowerShell:  $env:PREDICT_BACKEND="live"; python -m uvicorn suwon_chat.api:app --port 8000
```

- `live_check`는 칸마다 저장된 예측과 실시간 예측 사이의 거리를 재서 세 가지를 본다: 어댑터 예측이 같은가, 학습 전 모델 예측이 같은가, 맞는 어댑터가 끼워졌는가. "실패"가 나오면 실시간 버전을 시연에 쓰지 않는다. 정리본은 GPU 종류만 달라도 칸당 오차 평균이 0.0025쯤 달라진다고 적었고, 통과 기준은 칸별 거리 평균 0.005다. 4비트로 올리면 기준을 넘을 수 있다.
- 서버는 모델을 다 올린 뒤에 뜬다. 올리지 못하면 이유와 대처법을 적고 멈춘다. 장치와 메모리는 가중치를 읽기 전에 검사한다.
- GPU를 찾지 못하면 경고를 내고 CPU로 돌린다. 윈도우에서 `pip install -r requirements.txt`로 깔리는 torch는 CPU용이라, NVIDIA GPU가 있어도 CUDA용 torch를 따로 깔아야 GPU를 쓴다.
- 허깅페이스에서 받는 판은 `LIVE_BASE_REVISION`으로 고정돼 있다(어댑터를 학습한 가중치). 인터넷이 없는 곳에서는 미리 받아 둔 뒤 `HF_HUB_OFFLINE=1`을 주거나 `LIVE_BASE_MODEL`에 받아 둔 폴더를 적는다.
- 서버가 뜨면 추천 질문의 장면을 뒤에서 미리 돌려 둔다.
- 한 장면을 처음 볼 때 어댑터, 학습 전 모델, (집단·문항 제외 어댑터면) 전체 학습 어댑터를 차례로 돌린다. 같은 장면을 다시 보면 저장해 둔 값을 쓴다.
- 새 문항은 채팅에서 문항 문구와 보기(2~11개)를 함께 주면 된다. '잘 모르겠다' 같은 척도 밖 보기는 평균에서 뺀다. 예: `'수원시에 AI 행정 서비스가 도입되면 이용하시겠습니까?' 보기는 전혀 이용하지 않겠다, 이용하지 않는 편이다, 보통이다, 이용하는 편이다, 적극 이용하겠다. 연령별로 예측해 줘`

**이 버전은 실제 모델로 돌려 확인하지 못했다.** 만든 PC에 GPU가 없고 RAM이 7GB여서다. 확인한 것은 (1) 프롬프트가 정리본의 평가용 프롬프트 11,187개와 글자 하나까지 같다는 것, (2) 실제 Bllossom 토크나이저에서 보기 글자 토큰이 정리본의 계산과 같게 잡힌다는 것, (3) 계산 순서와 정밀도(bfloat16 로짓, float32 softmax)가 정리본의 코드·저장된 값과 같다는 것(독립 리뷰), (4) 무작위 가중치의 작은 모델로 모델 올리기·어댑터 갈아 끼우기·확률 계산·서버·화면이 끝까지 돈다는 것이다. 실제 값이 맞는지는 3번의 `live_check`가 처음으로 확인한다.

작은 모델로 배선만 확인하려면: `python tests/tiny.py /tmp/tiny` 뒤 `LIVE_BASE_MODEL=/tmp/tiny/base SUWON_ADAPTER_DIR=/tmp/tiny/adapters LIVE_DTYPE=float32 PREDICT_BACKEND=live python -m uvicorn suwon_chat.api:app --port 8010` (예측값은 뜻이 없다).

## 참고 문서

`reports/*.md`는 시연용으로 만든 가상 보고서다. 숫자는 2023·2024년 수원서베이 실측에서만 가져왔고, 예측에는 쓰이지 않는다.

- 형식: 머리말(`id`, `title`, `summary`, `topics`, `questions`)과 `<!-- page N -->`으로 나눈 쪽. 제목이 `[시연용 가상 문서]`로 시작하면 화면에 가상 문서로 표시된다.
- 문서를 고치거나 더하면 `python -m suwon_chat.reports`로 색인을 다시 만들고 서버를 다시 켠다.
- 검색은 로컬 임베딩 모델(`intfloat/multilingual-e5-small`)로 한다.
- 예측 옆에 띄울 문서는 두 가지를 겹쳐 고른다. 머리말의 `questions`에 그 문항이 적힌 문서가 있으면 그 문서를 띄우고(`"q189"`는 그 변수, `"q132_"`는 q132_로 시작하는 변수 모두), 쪽은 임베딩으로 가까운 것을 고른다. 어느 문서에도 적히지 않은 문항은 임베딩 유사도가 0.91을 넘는 문서가 있을 때만 띄운다. 임베딩만 썼을 때 무관한 문서가 뜨거나 정작 그 항목을 다룬 쪽이 빠지는 일이 있어서다.
- "관련 보고서 있어?"처럼 사용자가 직접 물을 때의 검색은 임베딩만 쓰고, 유사도 0.87 미만이면 관련 문서가 없다고 답한다 (`backend/suwon_chat/reports.py`의 문턱 값들).
- 가상 보고서를 쓸 때는 `python scripts/stats_2324.py`로 숫자를 찾는다. 2025년 자료는 보여 주지 않는다.

## 저장된 답변

대화 모델을 쓸 수 없을 때(키 없음, 인터넷 끊김) 추천 질문은 `backend/suwon_chat/saved_answers.json`의 글을 대신 내보내고 화면에 "저장된 답변"이라고 표시한다. 차트와 참고 문서는 그때 다시 계산한다.

```bash
python scripts/save_answers.py     # 추천 질문을 실제로 돌려 새로 저장한다 (요금이 든다). 저장한 뒤 사람이 읽고 고친다
```

끊긴 상황을 미리 보려면 `ANTHROPIC_BASE_URL=http://127.0.0.1:9 scripts/serve.sh 8001`로 서버를 하나 더 띄운다.

## 설정

| 환경 변수 | 기본값 | 뜻 |
|---|---|---|
| `ANTHROPIC_API_KEY` | (없음) | 대화 모델 키. 없으면 채팅만 꺼지고 패널은 돈다 |
| `CHAT_MODEL` | `claude-haiku-5-5` | 대화 모델. 값만 바꾸면 상위 모델로 돈다 |
| `CHAT_EFFORT` | `low` | 사고 깊이 (`low` · `medium` · `high`) |
| `SUWON_SOURCE_DIR` | 정리본 폴더가 있으면 그것, 없으면 `./source_data` | 원본 자료 폴더 |
| `SUWON_DB_PATH` | `./data/suwon.db` | 색인 DB |
| `SUWON_REPORTS_DIR` | `./reports` | 참고 문서 폴더 |
| `PREDICT_BACKEND` | `lookup` | `lookup` 저장된 예측 조회 · `live` 모델을 직접 돌림 |
| `LIVE_BASE_MODEL` | `MLP-KTLim/llama-3-Korean-Bllossom-8B` | 기본 모델. 미리 받아 둔 폴더 경로도 된다 |
| `SUWON_ADAPTER_DIR` | `./adapters` | `exp1` · `exp2` · `exp3` 어댑터 폴더가 든 곳 |
| `LIVE_DEVICE` · `LIVE_DTYPE` · `LIVE_4BIT` · `LIVE_BATCH` | (자동) | 장치(`cuda`·`mps`·`cpu`), 정밀도, 4비트 여부, 한 번에 돌릴 프롬프트 수 |

## 폴더

```
backend/suwon_chat/
  build_index.py   정리본 → 로컬 DB, 자체 점검
  store.py         DB를 읽어 메모리에 올린다
  metrics.py       오차(WD·TV)와 평균. WD는 정리본의 함수를 그대로 옮겼다
  predict.py       예측 백엔드 (지금은 저장된 예측을 조회하는 것 하나)
  service.py       화면과 대화 LLM에 줄 자료. 화면의 숫자는 모두 여기서 나온다
  scenarios.json   추천 질문 (이 파일만 고치면 버튼이 바뀐다)
  prompt.py        모델에 주는 프롬프트 (학습·평가에 쓴 형식 그대로)
  live.py          실시간 예측 백엔드 (Bllossom + LoRA 어댑터),  live_check.py  저장된 예측과 비교
  reports.py       참고 문서 색인과 검색 (로컬 임베딩)
  saved.py         저장된 답변 내보내기,  saved_answers.json  저장된 답변
  chat.py          대화 LLM (Claude API)과 도구
  api.py           앱 서버
frontend/src/      화면 (React + TypeScript). 차트는 라이브러리 없이 직접 그렸다
source_data/       앱이 읽는 원본 자료 (정리본에서 필요한 파일만 고른 것)
adapters/          LoRA 어댑터 세 개 (exp1 · exp2 · exp3)
reports/           시연용 가상 보고서 12편
scripts/           서버 실행, 답변 확인·저장, 2023·2024년 실측 조회
docs/PRD.md        요구사항
```
