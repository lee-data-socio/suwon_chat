# 수원서베이 응답 예측 챗봇 (시연용)

수원서베이(수원시 시민 대상 연례 설문)로 언어 모델을 학습시켜, **어떤 집단이 어떤 문항에 어떻게 답할지를 응답 분포로 예측**한 실험의 결과를 채팅으로 보여 주는 시연용 챗봇이다.

질문을 하면 답변 안에 예측과 실제 조사 결과를 나란히 놓은 차트가 들어가고, 연구 보고서(시연용 가상 문서)의 내용을 인용해 그 수치를 풀이해 준다.

> 실제 서비스가 아니라 실험 결과를 설명하기 위한 시연 화면이다. **처음 보는 분은 [소개 문서](docs/OVERVIEW.md)부터** 읽으면 된다. 이상적인 구조와 지금의 구조, 실행 방식을 쉬운 말로 정리했다. 이 README는 설치와 설정의 세부 사항을 다룬다.

## 무엇을 보여 주나

- **실험**: 한국어 언어 모델 Bllossom 8B에 작은 어댑터(LoRA)를 붙여 2023·2024년 수원서베이로 학습시키고, 2025년 조사 결과를 보지 않은 채 2025년 문항 113개의 응답 분포를 집단별로 예측했다. 방법은 SubPOP(UC Berkeley)을 따랐다.
- **화면**: 예측과 2025년 실측을 집단별 막대로 나란히 보여 주고, 얼마나 맞았는지를 "7점 척도에서 약 0.2점"처럼 읽기 쉬운 말로 알려 준다.
- **어댑터 세 개**를 갈아 끼워 볼 수 있다.

| 어댑터 | 학습 자료 | 보여 주는 것 |
|---|---|---|
| 전체 학습 어댑터 | 2023·2024년 전체 | 다음 해 조사 결과를 얼마나 맞히나 |
| 집단 제외 어댑터 | 7개 집단의 응답을 뺀 것 | 한 번도 보지 못한 집단을 얼마나 맞히나 |
| 문항 제외 어댑터 | 4개 문항을 뺀 것 | 한 번도 보지 못한 문항을 얼마나 맞히나 |

- **집단**: 성별, 연령, 주택유형, 점유형태, 고용형태, 통근·통학, 거주기간, 종교의 여덟 속성, 33개 집단. 한 번에 속성 하나로만 나눈다.
- **참고 문서**: `reports/`의 가상 보고서 12편. 답변이 이 문서의 내용(2023·2024년 조사 결과)을 제목·쪽수와 함께 인용해 수치를 해석한다. 화면에는 "시연용 가상 문서"로 표시된다. 예측 모델은 이 문서를 보지 않았다.

이 챗봇이 **하지 못하는 것**도 물으면 그렇게 답한다: 두 속성을 겹친 집단("30대 여성"), 연도에 따른 변화("내년에는?"), 모델이 그렇게 예측한 이유.

## 빨리 시작하기

저장소만 받으면 된다. 앱이 읽는 자료(`source_data/`)와 어댑터(`adapters/`)가 들어 있다. Python 3.12와 Node가 필요하다.

```bash
# 1. 파이썬 환경
conda create -n llm python=3.12 && conda activate llm
pip install -r requirements.txt

# 2. 대화 모델의 API 키 (없어도 된다. 그때는 추천 질문만 저장된 답변으로 나온다)
cp .env.example .env               # ANTHROPIC_API_KEY= 뒤에 키를 적는다

# 3. 자료 색인 (처음 한 번)
cd backend
python -m suwon_chat.build_index   # 예측과 실측 → data/suwon.db. 끝에 자체 점검을 돈다
python -m suwon_chat.reports       # 참고 문서 색인. 임베딩 모델(약 470MB)을 처음 한 번 내려받는다
python -m suwon_chat.concepts      # 문항 검색 색인 (개념 사전 concepts/개념사전.md + 문항)

# 4. 화면 빌드 (처음 한 번)
cd ../frontend
npm install && npm run build

# 5. 서버 실행
cd ../backend
python -m uvicorn suwon_chat.api:app --port 8000
```

브라우저에서 http://localhost:8000 을 연다. 리눅스·맥에서는 3번과 5번을 `scripts/serve.sh`가 한 번에 한다(색인이 없으면 둘 다 만든 뒤 서버를 띄운다).

## 써 보기

처음 화면의 추천 질문을 차례로 눌러 보면 실험의 줄거리를 따라갈 수 있다.

| 추천 질문 | 보여 주는 것 |
|---|---|
| 담당자처럼 묻기 · 형편 | 조사표 문구 없이 물어도 맞는 문항(경제·사회적 위치)을 찾고, 아직 안 한 조사를 예측하고, 지난 조사와 문서로 풀이하나 |
| 담당자처럼 묻기 · 시청 신뢰 | 단어 검색은 엉뚱한 문항(버스)을, 개념 사전은 공공기관 신뢰 문항을 찾는다. '질문 이해' 칸에서 차이가 보인다 |
| 이 예측을 믿어도 될까 | 2025년 조사를 가려 두고 맞혀 본 검증 결과. 작년 결과를 그대로 쓰는 것보다 나았나 |
| 조사에서 빠진 집단 | 한 집단(20년 이상 거주자)의 응답 없이 학습해도 그 집단을 예측하나 |
| 처음 묻는 문항 | 한 번도 학습하지 않은 문항(버스 인프라)을 처음 조사한다고 치고 예측하나 |
| 할 수 없는 것 | "30대 여성"처럼 할 수 없는 질문에 어떻게 답하나 |

이 시연은 2025년 조사를 **아직 하지 않았다고 놓는다**. 답변과 결과 카드는 예측과 지난 조사(2024년)만 보여 주고, 2025년 조사 결과는 카드의 **검증 보기**와 상세 패널에서만 "검증용"으로 보인다. 답변도 예측을 얼마나 믿을 수 있는지 물을 때만 검증 결과를 말한다.

'비교해서 보기'를 "일반 AI와"로 켜고 앞의 두 질문을 누르면, 같은 질문에 일반 AI가 어떻게 답하는지 옆 칸에 함께 나온다.

직접 물어도 된다.

- `수원시민 자부심을 거주기간별로 보여 줘`
- `연령별로 바꿔서 보여 줘` (보던 문항을 이어받는다)
- `이 결과와 관련된 보고서가 있어?`
- `반려동물이 있는지 연령별로 보여 줘` (예/아니오 문항)

왼쪽 **대화 기록**에서 지난 대화를 다시 열거나 새 대화를 시작한다. 기록은 서버 메모리에만 있어서 서버를 끄면 사라진다(파일로 남기지 않는다).

답변 속 결과 카드의 **자세히 보기**를 누르면 오른쪽에 상세 패널이 열린다. 문항·속성·어댑터를 직접 바꾸고, 작년 실측이나 학습 전 모델을 함께 볼 수 있다. 참고 문서 카드를 누르면 문서 전체가 열린다.

## 예측을 가져오는 세 가지 방식

`.env`의 `PREDICT_BACKEND` 값 하나로 고른다. 화면과 대화는 셋 다 같다.

| `PREDICT_BACKEND` | 하는 일 | 필요한 것 | 화면 표시 | 값 |
|---|---|---|---|---|
| `lookup` (기본) | 연구실 GPU 서버에서 미리 계산해 둔 예측을 `data/suwon.db`에서 꺼낸다 | 없음 | 사전 계산 | 보고서의 숫자 그대로 |
| `live` | GPU에서 Bllossom 8B를 원래 정밀도로 올려 직접 추론한다 | GPU 메모리 16GB 이상 | 실시간 | 저장된 예측과 거의 같아야 한다 (아래 "확인 상태") |
| `gguf` | CPU에서 4비트로 양자화한 Bllossom 8B로 직접 추론한다 | RAM 8GB 이상 | 실시간 · 양자화 | 저장된 예측과 꽤 다르다. 흐름을 돌려 보는 용도 |

`live`와 `gguf`는 `adapters/`의 어댑터 세 개를 갈아 끼우며 질문마다 추론한다. 목록에 없는 **새 문항**도 채팅에 문구와 보기를 주면 예측한다(실측이 없으므로 "실험적"으로 표시되고 오차는 나오지 않는다).

```
'수원시에 AI 행정 서비스가 도입되면 이용하시겠습니까?' 보기는 전혀 이용하지 않겠다, 이용하지 않는 편이다,
보통이다, 이용하는 편이다, 적극 이용하겠다. 연령별로 예측해 줘
```

**보고서의 숫자와 나란히 보여 줄 때는 `lookup`을 쓴다.**

### 확인 상태 (2026-11-15)

| 방식 | 확인한 것 | 확인하지 못한 것 |
|---|---|---|
| `lookup` | 113문항 × 8속성 × 3어댑터의 모든 값이 원본과 일치 (독립 검증 2회) | — |
| `live` | 프롬프트가 평가용 프롬프트 11,187개와 글자 단위로 같음. 계산 순서와 정밀도가 원본 코드와 같음(독립 리뷰). 작은 무작위 모델로 전체 경로가 돎 | **실제 모델로 돌려 보지 못했다** (만든 PC에 GPU가 없다). GPU가 있는 PC에서 `live_check`가 처음 확인한다 |
| `gguf` | 이 PC(GPU 없음, RAM 7.8GB)에서 실제로 돎. 올리는 데 16초, 프롬프트 하나에 약 6초. 맞는 어댑터가 끼워짐 | 값이 저장된 예측과 많이 다르다. 문항 × 속성 8쌍(34칸)에서 저장된 예측과의 거리 평균 0.079. 같다고 보는 기준(0.005)의 16배이고, 서로 다른 어댑터끼리의 거리(0.018)의 4배쯤이다. 어댑터를 바꿔 끼웠을 때의 차이보다 양자화 버전의 어긋남이 더 크다는 뜻이므로, 어댑터 비교에도 쓰지 않는다. 양자화 때문인지 다른 원인이 남았는지는 가리지 못했다 |

### GPU에서 원래 정밀도로 (`live`)

```bash
# NVIDIA GPU가 있으면 먼저 그 PC의 CUDA에 맞는 torch를 깐다 (pytorch.org의 설치 명령)
cd backend
python -m suwon_chat.live_check          # 빠른 점검: 어댑터마다 문항 × 속성 24쌍. 모델(약 16GB)을 처음 내려받는다
python -m suwon_chat.live_check --all    # 전부. 시연에 쓰기 전에 한 번은 돌린다
PREDICT_BACKEND=live python -m uvicorn suwon_chat.api:app --port 8000
#   윈도우 PowerShell:  $env:PREDICT_BACKEND="live"; python -m uvicorn suwon_chat.api:app --port 8000
```

| 장비 | 올리는 방법 |
|---|---|
| NVIDIA GPU, 메모리 16GB 이상 | 그대로 |
| NVIDIA GPU, 메모리 8~12GB | `LIVE_4BIT=1` (`pip install bitsandbytes`). 값이 조금 달라진다 |
| Apple Silicon(`mps`) · Intel GPU(`xpu`), 메모리 32GB 이상 | 그대로. 돌려 보지 못했다 |
| GPU 없음 | `gguf`를 쓴다 |

- `live_check`는 칸마다 저장된 예측과의 거리를 재서 세 가지를 본다: 어댑터 예측이 같은가, 학습 전 모델 예측이 같은가, 맞는 어댑터가 끼워졌는가. "실패"가 나오면 시연에 쓰지 않는다.
- 서버는 모델을 다 올린 뒤에 뜬다. 장치와 메모리는 가중치를 읽기 전에 검사하고, 문제가 있으면 대처법과 함께 알린다.
- 윈도우에서 `pip install -r requirements.txt`로 깔리는 torch는 CPU용이다. NVIDIA GPU가 있어도 CUDA용 torch를 따로 깔아야 GPU를 쓴다.
- 인터넷이 없는 곳에서는 모델을 미리 받아 둔 뒤 `HF_HUB_OFFLINE=1`을 준다.

### GPU 없이 양자화 모델로 (`gguf`)

```bash
pip install llama-cpp-python==0.3.36 --only-binary :all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
cd backend
python -m suwon_chat.quantized --download        # 양자화 모델(약 4.9GB)을 models/ 로 받는다
python -m suwon_chat.live_check --backend gguf --pairs 8 --adapter exp1    # 저장된 예측과 얼마나 다른지 잰다 (약 5분)
PREDICT_BACKEND=gguf python -m uvicorn suwon_chat.api:app --port 8000
```

- 모델은 Bllossom 팀이 올린 4비트 파일이다. RAM이 넉넉하면 `GGUF_REPO`·`GGUF_FILE`로 8비트 파일(8.5GB)을 고를 수 있다. 값이 더 가까워지는지는 재 보지 못했다.
- 한 장면을 처음 볼 때는 집단마다 어댑터와 학습 전 모델을 한 번씩 돌린다(집단 3개면 약 40초, 7개면 1분 반). 다시 볼 때는 바로 나온다. 서버가 뜨면 추천 질문의 장면을 뒤에서 미리 돌려 둔다.
- 어댑터는 llama.cpp 형식으로 바꿔 둔 `adapters/exp*/adapter.gguf`를 쓴다. 새로 바꿀 때는 `pip install gguf` 뒤 `python scripts/convert_adapters_gguf.py`.

## 대화 모델

답변 문장은 Claude API가 쓴다. 숫자는 대화 모델이 만들지 않고, 앱 서버가 계산한 값을 도구로 받아 쓴다. 차트의 숫자는 대화 모델을 거치지 않는다.

- 기본 모델은 요금이 가장 낮은 `claude-haiku-5-5`다. `.env`의 `CHAT_MODEL` 값만 바꾸면 상위 모델로 돈다.
- **키가 없거나 인터넷이 끊기면** 추천 질문은 미리 저장해 둔 답변(`backend/suwon_chat/saved_answers.json`)으로 나오고 화면에 "저장된 답변"이라고 표시된다. 차트와 참고 문서는 그대로 나온다. 직접 입력한 질문은 안내 문구로 끝난다.
- 저장된 답변을 새로 받으려면 서버를 띄운 채 `python scripts/save_answers.py`를 돌린다(요금이 든다). 받은 뒤에는 사람이 읽고 다듬는다.
- 끊긴 상황을 미리 보려면 `ANTHROPIC_BASE_URL=http://127.0.0.1:9 python -m uvicorn suwon_chat.api:app --port 8001`.

## 질문 이해 (개념 사전)

시청 담당자는 조사표 문구로 묻지 않는다("월세 사는 사람들 형편 어때?"). `concepts/개념사전.md`는 문항마다 조사가 재는 개념, 그 정의, 담당자가 쓸 법한 일상 표현을 적은 문서이고, 문항 검색은 이 사전을 덧붙인 의미 검색이다(로컬 임베딩).

- 대화 모델에는 문항 목록을 주지 않는다. 문항은 `search_questions`로만 찾는다. 작은 로컬 모델은 문항 목록 전체를 받아 판단하기 어렵기 때문이다.
- 답변 위의 **질문 이해** 칸은 같은 질문을 단어 검색(RAG 없음)과 개념 사전 검색(RAG)으로 각각 찾은 문항 후보를 나란히 보인다. 답변에 쓴 문항은 표시된다.
- 근거 단계 `data`는 단어 검색만, `full`은 개념 사전 검색을 쓴다.
- 사전을 고치면 `python -m suwon_chat.concepts`로 색인을 다시 만든다.
- 담당자식 질문 40개로 검색 방식을 견준다: `cd backend && python ../scripts/eval_question_search.py --show`. 1순위 적중은 단어 검색 8/20·8/20, 개념 사전 검색 16/20·19/20이다(2026-11-20).

## 참고 문서

`reports/*.md`는 시연용으로 만든 가상 보고서 12편이다. 숫자는 2023·2024년 수원서베이 실측에서만 가져왔다(12편의 숫자 949개를 실측과 대조했다).

- 형식: 머리말(`id`, `title`, `summary`, `topics`, `questions`)과 `<!-- page N -->`으로 나눈 쪽. 제목이 `[시연용 가상 문서]`로 시작하면 화면에 가상 문서로 표시된다.
- 예측을 보여 줄 때는 머리말의 `questions`에 그 문항이 적힌 문서를 띄우고(`"q189"`는 그 변수, `"q132_"`는 q132_로 시작하는 변수 모두), 대화 모델에 그 문서를 주어 해석에 인용하게 한다. "관련 보고서 있어?"처럼 직접 물을 때는 임베딩 검색(`intfloat/multilingual-e5-small`, 로컬)으로 찾는다.
- 문서를 고치거나 더하면 `python -m suwon_chat.reports`로 색인을 다시 만들고 서버를 다시 켠다.
- 가상 보고서를 쓸 때는 `python scripts/stats_2324.py`로 숫자를 찾는다. 2025년 자료는 보여 주지 않는다.

## 테스트

```bash
cd backend
python -m pytest -q tests          # API 키도 GPU도 필요 없다
```

- 앱이 계산한 칸 단위 오차를 정리본 결과표의 값과 견준다.
- 실시간 버전의 프롬프트를 평가용 프롬프트와 글자 단위로 견주고, 모델 올리기·어댑터 교체·확률 계산을 작은 무작위 모델로 확인한다.
- 실제 대화 모델의 답변은 서버를 띄운 채 `python scripts/check_answers.py 3`으로 본다(요금이 든다). 추천 질문을 3번씩 돌려 답변 속 숫자가 도구 결과나 참고 문서에 있는 값인지 확인한다.

## 설정 (`.env` 또는 환경 변수)

`.env.example`을 `.env`로 복사해 값을 적는다. 자주 쓰는 항목은 그 파일에 설명과 함께 들어 있고, 적지 않은 항목은 아래 기본값으로 돈다. 값을 바꾼 뒤에는 서버를 다시 켠다.

| 이름 | 기본값 | 뜻 |
|---|---|---|
| `ANTHROPIC_API_KEY` | (없음) | 대화 모델 키 |
| `CHAT_MODEL` · `CHAT_EFFORT` | `claude-haiku-5-5` · `low` | 대화 모델과 사고 깊이 |
| `PREDICT_BACKEND` | `lookup` | `lookup` · `live` · `gguf` |
| `LIVE_BASE_MODEL` · `LIVE_DEVICE` · `LIVE_DTYPE` · `LIVE_4BIT` · `LIVE_BATCH` | Bllossom 8B · (자동) | `live`의 모델, 장치(`cuda`·`mps`·`xpu`·`cpu`), 정밀도, 4비트 여부, 묶음 크기 |
| `GGUF_MODEL` · `GGUF_REPO` · `GGUF_FILE` · `GGUF_THREADS` | `./models/…Q4_K_M.gguf` | `gguf`의 모델 파일, 받을 곳, CPU 스레드 수 |
| `SUWON_ADAPTER_DIR` | `./adapters` | 어댑터 폴더 |
| `SUWON_SOURCE_DIR` | 정리본 폴더가 있으면 그것, 없으면 `./source_data` | 원본 자료 폴더 |
| `SUWON_DB_PATH` · `SUWON_REPORTS_DIR` | `./data/suwon.db` · `./reports` | 색인 DB, 참고 문서 폴더 |

## 폴더

```
backend/suwon_chat/
  build_index.py   원본 자료 → 로컬 DB, 자체 점검
  store.py         DB를 읽어 메모리에 올린다
  metrics.py       오차(WD·TV)와 평균. WD는 정리본의 함수를 그대로 옮겼다
  service.py       화면과 대화 모델에 줄 자료. 화면의 숫자는 모두 여기서 나온다
  predict.py       조회 백엔드 (저장된 예측)
  prompt.py        모델에 주는 프롬프트 (학습·평가에 쓴 형식 그대로)
  live.py          실시간 백엔드 (GPU, transformers + PEFT)
  quantized.py     양자화 백엔드 (CPU, llama.cpp)
  live_check.py    실시간·양자화 예측을 저장된 예측과 비교
  reports.py       참고 문서 색인과 검색 (로컬 임베딩)
  chat.py          대화 모델(Claude API)과 도구
  saved.py         저장된 답변,  saved_answers.json · scenarios.json
  api.py           앱 서버
frontend/src/      화면 (React + TypeScript). 차트는 라이브러리 없이 직접 그렸다
source_data/       앱이 읽는 원본 자료 (정리본에서 필요한 파일만 고른 것)
adapters/          LoRA 어댑터 세 개 (safetensors와 llama.cpp용 gguf)
reports/           시연용 가상 보고서 12편
scripts/           서버 실행, 답변 확인·저장, 어댑터 변환, 자료 묶음 만들기
docs/OVERVIEW.md   처음 보는 분을 위한 소개 (이상적인 구조, 지금의 구조, 실행 방식)
```

## 자료와 출처

- 설문 자료는 수원서베이 2023~2025년의 **집단별 집계**다. 개인 단위 응답은 들어 있지 않다.
- 기본 모델: [MLP-KTLim/llama-3-Korean-Bllossom-8B](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)
- 학습 방법: SubPOP (UC Berkeley Canny Lab)
