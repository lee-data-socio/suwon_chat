"""저장된 답변. 추천 질문의 답변을 미리 받아 검수해 두고, 대화 모델을 쓸 수 없을 때 대신 내보낸다.

글만 저장한다. 차트 자료와 참고 문서는 내보낼 때 다시 계산하므로 화면의 숫자는 늘 지금 자료와 같다.
저장된 답변은 화면에 "저장된 답변"이라고 표시된다.

    python scripts/save_answers.py      # 추천 질문을 실제로 돌려 saved_answers.json을 새로 만든다 (그 뒤 사람이 읽고 고친다)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from .service import NotFound, Service

PATH = Path(__file__).with_name("saved_answers.json")
CHUNK = 12            # 한 번에 내보내는 글자 수 (받아 적는 것처럼 보이게)
DELAY = 0.02          # 조각 사이 간격(초)
# 이 오류로 대화가 시작도 못 했을 때 저장된 답변으로 넘어간다
FALLBACK_CODES = {"no_api_key", "auth", "connection", "api_error", "server_error"}


def load(path: Path = PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


async def replay(saved: dict, service: Service, reports, delay: float = DELAY) -> AsyncIterator[tuple[str, dict]]:
    """저장된 답변을 대화와 같은 이벤트로 내보낸다."""
    yield "saved", {"saved_at": saved.get("saved_at"), "model": saved.get("model")}
    panel = saved.get("panel")
    if panel:
        try:
            prediction = service.prediction(panel["qkey"], panel["attribute"], panel["adapter"])
        except NotFound:
            prediction = None       # 저장한 뒤 문항 목록이 바뀌었다. 글만 내보낸다
        if prediction:
            state = {k: panel[k] for k in ("qkey", "attribute", "adapter")}
            yield "panel", {"state": state, "compare": panel.get("compare", "actual"), "prediction": prediction}
            if reports and reports.available:
                yield "context", {"qkey": panel["qkey"], "query": prediction["question"]["text"],
                                  "items": reports.for_question(panel["qkey"])}
    text = saved["text"]
    for i in range(0, len(text), CHUNK):
        yield "text", {"delta": text[i:i + CHUNK]}
        if delay:
            await asyncio.sleep(delay)
    yield "done", {"stop_reason": "saved"}
