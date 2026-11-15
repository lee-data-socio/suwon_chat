# 수원서베이 시연 챗봇

수원서베이 응답 분포 예측 실험의 결과를 채팅과 차트로 보여 주는 시연 화면이다.
무엇을 왜 만드는지는 [docs/PRD.md](docs/PRD.md)에 있다.

## 준비

원본 자료 폴더 `수원서베이_학습결과_정리본/`이 이 폴더 바로 아래에 있어야 한다 (git에는 올리지 않는다).

```bash
conda activate llm                 # Python 3.12
pip install -r requirements.txt
cp .env.example .env               # ANTHROPIC_API_KEY를 채운다. 없어도 패널은 돈다
```

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

## 참고 문서

`reports/*.md`는 시연용으로 만든 가상 보고서다. 숫자는 2023·2024년 수원서베이 실측에서만 가져왔고, 예측에는 쓰이지 않는다.

- 형식: 머리말(`id`, `title`, `summary`, `topics`)과 `<!-- page N -->`으로 나눈 쪽. 제목이 `[시연용 가상 문서]`로 시작하면 화면에 가상 문서로 표시된다.
- 문서를 고치거나 더하면 `python -m suwon_chat.reports`로 색인을 다시 만들고 서버를 다시 켠다.
- 검색은 로컬 임베딩 모델(`intfloat/multilingual-e5-small`)로 한다. 유사도가 0.87보다 낮으면 관련 문서가 없다고 보고 띄우지 않는다(`backend/suwon_chat/reports.py`의 `MIN_SCORE`).
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
| `SUWON_SOURCE_DIR` | `./수원서베이_학습결과_정리본` | 원본 자료 폴더 |
| `SUWON_DB_PATH` | `./data/suwon.db` | 색인 DB |
| `SUWON_REPORTS_DIR` | `./reports` | 참고 문서 폴더 |

## 폴더

```
backend/suwon_chat/
  build_index.py   정리본 → 로컬 DB, 자체 점검
  store.py         DB를 읽어 메모리에 올린다
  metrics.py       오차(WD·TV)와 평균. WD는 정리본의 함수를 그대로 옮겼다
  predict.py       예측 백엔드 (지금은 저장된 예측을 조회하는 것 하나)
  service.py       화면과 대화 LLM에 줄 자료. 화면의 숫자는 모두 여기서 나온다
  scenarios.json   추천 질문 (이 파일만 고치면 버튼이 바뀐다)
  reports.py       참고 문서 색인과 검색 (로컬 임베딩)
  saved.py         저장된 답변 내보내기,  saved_answers.json  저장된 답변
  chat.py          대화 LLM (Claude API)과 도구
  api.py           앱 서버
frontend/src/      화면 (React + TypeScript). 차트는 라이브러리 없이 직접 그렸다
reports/           시연용 가상 보고서 12편
scripts/           서버 실행, 답변 확인·저장, 2023·2024년 실측 조회
docs/PRD.md        요구사항
```
