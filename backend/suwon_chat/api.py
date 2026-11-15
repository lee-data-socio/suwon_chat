"""앱 서버.

    uvicorn suwon_chat.api:app --port 8000

화면이 쓰는 주소:
    GET  /api/meta         어댑터, 속성과 집단, 예측 출처, 색인 자체 점검 결과
    GET  /api/questions    2025년 문항 목록 (?q= 검색어)
    GET  /api/prediction   문항 × 속성 × 어댑터의 예측·실측·오차 (?qkey=&attribute=&adapter=)
    POST /api/predict_new  목록에 없는 새 문항의 실험적 예측 (실시간 버전에서만)
    GET  /api/scenarios    추천 질문
    GET  /api/context      문항과 가까운 참고 문서 대목 (?qkey=)
    GET  /api/reports      참고 문서 목록,  GET /api/reports/{doc_id}  문서 한 편의 쪽별 전문
    GET  /api/chat/status  대화 LLM을 쓸 수 있는지
    POST /api/chat         대화. 응답은 SSE(text/event-stream)이고 이벤트는
                           status {tool,text} · text {delta} · panel {state,compare,prediction} · new_prediction {prediction} · context {qkey,query,items} · done {stop_reason} · error {code,message}
                           · saved {saved_at,model} (대화 모델 대신 저장된 답변을 내보낼 때 맨 앞에 온다)
"""
from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config
from .chat import chat_status, run_chat
from .predict import LookupBackend
from . import saved
from .reports import LocalEmbedder, ReportIndex
from .scenarios import SCENARIOS
from .service import NEW_MAX_OPTIONS, Invalid, NotFound, Service, Unsupported
from .store import Store

# 이 앱의 안내(모델 올리기, 실시간 예측에 걸린 시간 등)가 서버 창에 보이게 한다
_app_log = logging.getLogger("suwon_chat")
if not _app_log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    _app_log.addHandler(_handler)
    _app_log.setLevel(logging.INFO)
    _app_log.propagate = False
log = logging.getLogger("suwon_chat.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = Store()
    if config.PREDICT_BACKEND == "live":
        from .live import LiveBackend
        backend = LiveBackend(store)
        backend.load()            # 모델을 못 올리면 여기서 이유와 함께 멈춘다. 반쯤 켜진 서버를 두지 않는다
    elif config.PREDICT_BACKEND == "gguf":
        from .quantized import QuantizedBackend
        backend = QuantizedBackend(store)
        backend.load()
    elif config.PREDICT_BACKEND == "lookup":
        backend = LookupBackend(store)
    else:
        raise RuntimeError(f"PREDICT_BACKEND는 lookup, live, gguf 가운데 하나여야 한다: {config.PREDICT_BACKEND!r}")
    app.state.service = Service(store, backend)
    if backend.source == "live":
        # 추천 질문의 장면을 미리 돌려 둔다. 발표에서 처음 누를 때 기다리지 않게 한다
        threading.Thread(target=_warm_scenarios, args=(app.state.service,), daemon=True).start()
    embedder = LocalEmbedder()
    app.state.reports = ReportIndex(config.REPORTS_INDEX, config.REPORTS_VECTORS, _guarded(embedder))
    if app.state.reports.available and config.WARM_EMBEDDER:
        # 첫 검색이 느리지 않게 임베딩 모델을 미리 올려 둔다. 서버 시작은 기다리게 하지 않는다.
        threading.Thread(target=_warm, args=(embedder,), daemon=True).start()
    yield


def _warm_scenarios(service: Service) -> None:
    for s in SCENARIOS:
        if s["panel"]:
            try:
                service.prediction(s["panel"]["qkey"], s["panel"]["attribute"], s["panel"]["adapter"])
            except Exception:
                log.warning("추천 질문 %s을 미리 돌리지 못했다", s["id"], exc_info=True)
    log.info("추천 질문의 장면을 미리 돌려 두었다")


def _warm(embedder: LocalEmbedder) -> None:
    try:
        embedder.load()
    except Exception:
        log.warning("임베딩 모델을 올리지 못했다. 문항별 참고 문서는 나오지만 자유 검색은 되지 않는다.", exc_info=True)


def _guarded(embedder: LocalEmbedder):
    """임베딩 모델을 못 쓰는 환경에서도 서버가 죽지 않게 한다. 검색 결과가 비어 나갈 뿐이다."""
    def embed(texts, kind):
        try:
            return embedder(texts, kind)
        except Exception:
            log.warning("임베딩 실패. 검색 결과를 비워 돌려준다.", exc_info=True)
            return np.zeros((len(texts), 1), np.float32)
    embed.model_name = embedder.model_name
    return embed


app = FastAPI(title="수원서베이 시연 챗봇", lifespan=lifespan)


def svc() -> Service:
    return app.state.service


def reports() -> ReportIndex:
    return app.state.reports


@app.get("/api/meta")
def meta():
    r = reports()
    backend = svc().backend
    return {**svc().meta(), "reports": {"available": r.available, "count": len(r.documents)},
            "can_predict_new": svc().can_predict_new(),
            "live": backend.describe() if hasattr(backend, "describe") else None}


@app.get("/api/context")
def context(qkey: str):
    q = svc().store.questions.get(qkey)
    if q is None or q.year != config.PREDICT_YEAR:
        raise HTTPException(404, f"{config.PREDICT_YEAR}년 문항이 아니다: {qkey}")
    r = reports()
    return {"available": r.available, "query": q.text, "items": r.for_question(qkey)}


@app.get("/api/reports")
def report_list():
    return reports().listing()


@app.get("/api/reports/{doc_id}")
def report(doc_id: str):
    doc = reports().document(doc_id)
    if doc is None:
        raise HTTPException(404, f"없는 문서: {doc_id}")
    return doc


@app.get("/api/questions")
def questions(q: str = ""):
    return svc().questions(q)


@app.get("/api/prediction")
def prediction(qkey: str, attribute: str, adapter: str = "exp1"):
    try:
        return svc().prediction(qkey, attribute, adapter)
    except NotFound as e:
        raise HTTPException(404, str(e))


class NewQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    options: list[str] = Field(min_length=2, max_length=NEW_MAX_OPTIONS + 1)      # 끝에 무응답을 붙여 보내도 받는다
    ordered: bool = True
    off_scale_options: list[str] = Field(default_factory=list, max_length=NEW_MAX_OPTIONS)
    attribute: str = Field(max_length=40)
    adapter: str = Field(default="exp1", max_length=40)


@app.post("/api/predict_new")
def predict_new(req: NewQuestion):
    try:
        return svc().new_prediction(req.question, req.options, req.ordered, req.attribute, req.adapter,
                                    req.off_scale_options)
    except Unsupported as e:
        raise HTTPException(409, str(e))
    except Invalid as e:
        raise HTTPException(422, str(e))
    except NotFound as e:
        raise HTTPException(404, str(e))


@app.get("/api/scenarios")
def scenarios():
    return SCENARIOS


@app.get("/api/chat/status")
def status():
    return chat_status()


# 길이 한도는 넉넉히 받고, 모델에 넘길 때 서버가 자른다 (chat.normalize_history).
# 한도를 넘는다고 요청을 거부하면 긴 대화가 그 뒤로 계속 막힌다.
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=100_000)


class PanelState(BaseModel):
    qkey: str = Field(max_length=40)
    attribute: str = Field(max_length=40)
    adapter: str = Field(max_length=40)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=400)
    panel: PanelState | None = None
    # 추천 질문 버튼으로 보낸 질문이면 그 id. 대화 모델을 쓸 수 없을 때 저장된 답변으로 넘어가는 데 쓴다.
    scenario_id: str | None = Field(default=None, max_length=40)


def _valid_panel(panel: PanelState | None) -> dict | None:
    """화면이 보낸 패널 상태가 실제로 있는 문항·속성·어댑터일 때만 대화 LLM에 넘긴다."""
    if panel is None:
        return None
    store = svc().store
    q = store.questions.get(panel.qkey)
    if q is None or q.year != config.PREDICT_YEAR or panel.attribute not in store.groups \
            or panel.adapter not in config.ADAPTERS:
        return None
    return panel.model_dump()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.messages[-1].role != "user" or not req.messages[-1].content.strip():
        raise HTTPException(400, "마지막 메시지는 비어 있지 않은 사용자 메시지여야 한다")
    panel = _valid_panel(req.panel)

    answer = saved.load().get(req.scenario_id) if req.scenario_id else None

    async def live():
        if not chat_status()["available"]:
            yield "error", {"code": "no_api_key", "message": "대화 모델이 연결되지 않았습니다. 추천 질문은 저장된 답변으로 볼 수 있습니다."}
            return
        import anthropic
        # 발표 중에 오래 매달리지 않게 한다. 연결 5초, 응답 60초, 재시도 1번.
        client = anthropic.AsyncAnthropic(timeout=anthropic.Timeout(60.0, connect=5.0), max_retries=1)
        try:
            async for item in run_chat(client, svc(), [m.model_dump() for m in req.messages], panel, reports()):
                yield item
        except Exception:
            # 어떤 실패든 화면이 기다리기만 하지 않도록 error 이벤트로 끝낸다
            log.exception("대화 처리 실패")
            yield "error", {"code": "server_error", "message": "답변을 만드는 중 문제가 생겼습니다. 다시 물어 주세요."}
        finally:
            await client.close()

    async def stream():
        wrote = False
        async for event, data in live():
            # 글을 한 자도 쓰기 전에 대화 모델이 막혔고 저장된 답변이 있으면 그것으로 넘어간다
            if event == "error" and not wrote and answer and data.get("code") in saved.FALLBACK_CODES:
                log.warning("대화 모델을 쓸 수 없어 저장된 답변을 내보낸다 (%s, %s)", req.scenario_id, data.get("code"))
                async for item in saved.replay(answer, svc(), reports()):
                    yield _sse(*item)
                return
            wrote = wrote or event == "text"
            yield _sse(event, data)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# 빌드한 화면(frontend/dist)을 같은 서버에서 내보낸다. 요청마다 파일을 찾으므로 화면을 다시 빌드해도
# 서버를 다시 켤 필요가 없다. (개발 중에는 Vite 개발 서버가 /api를 이쪽으로 넘긴다.)
@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path == "api" or path.startswith("api/"):
        raise HTTPException(404, "없는 주소")
    dist = config.FRONTEND_DIST
    index = dist / "index.html"
    if not index.is_file():
        raise HTTPException(404, "화면이 아직 빌드되지 않았다. frontend 폴더에서 `npm run build`를 돌린다.")
    file = dist / path
    if path and file.is_file() and file.resolve().is_relative_to(dist.resolve()):
        return FileResponse(file)
    if path.startswith("assets/"):
        raise HTTPException(404, "없는 파일")       # 없는 스크립트 요청에 index.html을 주면 원인을 찾기 어렵다
    return FileResponse(index, headers={"Cache-Control": "no-cache"})
