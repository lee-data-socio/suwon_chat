"""앱 서버.

    uvicorn suwon_chat.api:app --port 8000

화면이 쓰는 주소:
    GET  /api/meta         어댑터, 속성과 집단, 예측 출처, 색인 자체 점검 결과
    GET  /api/questions    2025년 문항 목록 (?q= 검색어)
    GET  /api/prediction   문항 × 속성 × 어댑터의 예측·실측·오차 (?qkey=&attribute=&adapter=)
    GET  /api/scenarios    추천 질문
    GET  /api/chat/status  대화 LLM을 쓸 수 있는지
    POST /api/chat         대화. 응답은 SSE(text/event-stream)이고 이벤트는
                           status {tool,text} · text {delta} · panel {state,prediction} · done {stop_reason} · error {code,message}
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .chat import chat_status, run_chat
from .predict import LookupBackend
from .scenarios import SCENARIOS
from .service import NotFound, Service
from .store import Store

log = logging.getLogger("suwon_chat.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = Store()
    app.state.service = Service(store, LookupBackend(store))
    yield


app = FastAPI(title="수원서베이 시연 챗봇", lifespan=lifespan)


def svc() -> Service:
    return app.state.service


@app.get("/api/meta")
def meta():
    return svc().meta()


@app.get("/api/questions")
def questions(q: str = ""):
    return svc().questions(q)


@app.get("/api/prediction")
def prediction(qkey: str, attribute: str, adapter: str = "exp1"):
    try:
        return svc().prediction(qkey, attribute, adapter)
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

    async def stream():
        state = chat_status()
        if not state["available"]:
            yield _sse("error", {"code": "no_api_key", "message": "대화 모델이 연결되지 않았습니다. 패널은 그대로 쓸 수 있습니다."})
            return
        import anthropic
        # 발표 중에 오래 매달리지 않게 한다. 연결 5초, 응답 60초, 재시도 1번.
        client = anthropic.AsyncAnthropic(timeout=anthropic.Timeout(60.0, connect=5.0), max_retries=1)
        try:
            async for event, data in run_chat(client, svc(), [m.model_dump() for m in req.messages], panel):
                yield _sse(event, data)
        except Exception:
            # 어떤 실패든 화면이 기다리기만 하지 않도록 error 이벤트로 끝낸다
            log.exception("대화 처리 실패")
            yield _sse("error", {"code": "server_error", "message": "답변을 만드는 중 문제가 생겼습니다. 패널은 그대로 쓸 수 있습니다."})
        finally:
            await client.close()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# 빌드한 화면이 있으면 같은 서버에서 내보낸다 (개발 중에는 Vite 개발 서버가 /api를 이쪽으로 넘긴다)
if (config.FRONTEND_DIST / "index.html").exists():
    if (config.FRONTEND_DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "없는 주소")
        file = config.FRONTEND_DIST / path
        if path and file.is_file() and file.resolve().is_relative_to(config.FRONTEND_DIST.resolve()):
            return FileResponse(file)
        return FileResponse(config.FRONTEND_DIST / "index.html")
