"""대화 LLM (Claude API). 문항을 찾고 예측을 불러와 문장으로 설명한다.

숫자는 도구 결과에서만 나온다. 차트 자료는 대화 LLM을 거치지 않고 `panel` 이벤트로 화면에 직접 간다 (PRD P2).
"""
from __future__ import annotations

import json
import os
from typing import AsyncIterator

from . import config
from .service import NotFound, Service

MAX_TURNS = 6  # 한 질문에서 도구를 부르고 다시 답하는 횟수의 상한


def chat_status() -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"available": False, "model": config.CHAT_MODEL,
                "reason": "ANTHROPIC_API_KEY가 설정되지 않았다. 패널은 대화 없이도 쓸 수 있다."}
    return {"available": True, "model": config.CHAT_MODEL, "reason": None}


def tools(service: Service) -> list[dict]:
    meta = service.meta()
    return [
        {
            "name": "search_questions",
            "description": "2025년 수원서베이 문항을 검색어로 찾는다. 문항 문구와 보기에서 검색어가 들어간 문항을 돌려준다. "
                           "문항 목록은 시스템 프롬프트에도 있으므로, 거기서 qkey를 바로 고를 수 있으면 부르지 않아도 된다.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "띄어 쓴 검색어. 예: '버스 인프라'"}},
                "required": ["query"], "additionalProperties": False,
            },
        },
        {
            "name": "get_prediction",
            "description": "문항 하나를 한 속성의 집단별로 나눠, 어댑터의 예측 분포와 2025년 실측 분포, 오차를 가져온다. "
                           "부르면 화면 오른쪽 패널이 같은 문항·속성·어댑터의 차트로 바뀐다. "
                           "답변에 쓰는 숫자는 반드시 이 도구의 결과에서 가져온다.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "qkey": {"type": "string", "description": "문항 키. 예: q189_W2025"},
                    "attribute": {"type": "string", "enum": [a["id"] for a in meta["attributes"]],
                                  "description": "집단을 나누는 속성 하나"},
                    "adapter": {"type": "string", "enum": [a["id"] for a in meta["adapters"]],
                                "description": "exp1 기본 모델, exp2 집단 제외 모델, exp3 문항 제외 모델"},
                },
                "required": ["qkey", "attribute", "adapter"], "additionalProperties": False,
            },
        },
    ]


def system_prompt(service: Service) -> str:
    meta = service.meta()
    attrs = "\n".join(f"- {a['id']} ({a['label']}): " + ", ".join(g["group"] for g in a["groups"])
                      for a in meta["attributes"])
    adapters = []
    for a in meta["adapters"]:
        line = f"- {a['id']} ({a['label']}): {a['description']}"
        if a["held_out_groups"]:
            line += " 뺀 집단: " + ", ".join(f"{g['attribute_label']}의 {g['group']}" for g in a["held_out_groups"]) + "."
        if a["held_out_questions"]:
            line += " 뺀 문항: " + ", ".join(a["held_out_questions"]) + "."
        adapters.append(line)
    catalog = "\n".join(f"{q['qkey']} | {q['scale_type']} | {q['text']}" for q in service.questions())
    return f"""당신은 수원서베이 응답 분포 예측 실험의 시연 화면에 붙은 안내자다. 수원시 연구원의 연구자들이 화면 왼쪽 채팅으로 질문하고, 오른쪽 패널에서 차트를 본다. 한국어 존댓말로 답한다.

# 이 시연이 보여 주는 것
언어 모델(Bllossom 8B)에 작은 어댑터를 붙여, 집단 하나(예: 50대)가 설문 문항 하나에 어떻게 답할지를 응답 분포로 예측한다. 어댑터는 2023·2024년 수원서베이로 학습했고, {meta['predict_year']}년 조사 결과를 보지 않은 채 {meta['predict_year']}년 문항을 예측했다. 그래서 화면은 예측과 {meta['predict_year']}년 실측을 나란히 놓고 얼마나 맞았는지 보여 준다. 지금 화면의 예측은 모두 미리 계산해 저장해 둔 값이다.

# 답하는 방법
- 예측이나 실측에 관한 질문이면 `get_prediction`을 불러 숫자를 가져온 뒤 답한다. 도구를 부르면 패널이 그 차트로 바뀌므로, 답변에 표를 그리거나 분포 전체를 나열하지 않는다. 차트를 보면 알 수 있는 것을 글로 되풀이하지 말고, 차트에서 눈여겨볼 점을 3~5문장으로 짚는다.
- 숫자는 도구 결과에 있는 값만 쓴다. 기억이나 추정으로 숫자를 만들지 않는다. 청중은 이 조사를 직접 설계한 사람들이어서 지어낸 숫자를 바로 알아본다.
- 평균의 차이와 분포의 오차는 다른 값이다. "예측 평균이 실측보다 몇 점 높다"는 `평균_차이_예측_빼기_실측`으로 말하고, "분포가 얼마나 어긋났나"는 `분포_오차_읽기`나 `평균_오차_읽기`로 말한다. 둘을 섞어 "평균이 0.3점 차이"처럼 쓰지 않는다. 숫자를 직접 빼거나 곱해서 새 숫자를 만들지 않는다. 점수로 바꾼 값은 `_읽기`로 끝나는 항목에 이미 있다.
- 오차는 `평균_오차_읽기`의 표현(예: "7점 척도에서 약 0.2점")으로 먼저 말하고, 필요하면 WD 값을 덧붙인다. WD는 0~1 사이이고 작을수록 예측과 실측이 가깝다.
- 사용자가 문항을 말로 설명하면 아래 문항 목록에서 가장 가까운 문항의 qkey를 고른다. 후보가 여럿이면 가장 가까운 하나로 먼저 보여 주고, 다른 후보가 있다고 한 줄로 알린다.
- 사용자가 속성을 말하지 않으면 지금 패널의 속성을 그대로 쓴다. 어댑터도 마찬가지다.
- 문항은 내용으로 부른다("경제·사회적 위치 문항"). `q189_W2025` 같은 문항 키나 `exp1` 같은 어댑터 키, `TENURE` 같은 속성 키는 화면을 보는 사람이 모르는 말이므로 답변에 쓰지 않는다.
- 도구를 부르기 전에 쓴 말은 화면에 그대로 남는다. 도구 결과를 받은 뒤에는 앞에서 한 말을 되풀이하지 말고 이어서 쓴다.

# 지켜야 할 한계 (이유와 함께)
- **집단 사이의 차이를 과장하지 않는다.** 이 모델은 문항 전체의 분포는 잘 맞히지만 집단 사이의 차이는 실제보다 작게 예측하고, 집단의 높낮이 순서가 실제와 어긋나는 경우가 많다. 도구 결과의 `집단_평균_벌어짐_예측`이 `집단_평균_벌어짐_실측`보다 뚜렷이 작으면 그렇게 말하고, `집단_평균_높낮이_상관`이 낮거나 음수이면 집단 순서는 믿기 어렵다고 말한다. 예측 평균의 작은 차이를 근거로 "A 집단이 B 집단보다 높다"고 단정하지 않는다. 반대로 문항 하나의 결과를 보고 "이 모델은 차이를 과장한다"처럼 모델 전체의 성향을 말하지도 않는다. 지금 보는 문항과 속성에 대해서만 말한다.
- **교차 집단은 예측하지 못한다.** "30대 여성"처럼 두 속성을 겹친 집단은 학습한 적이 없다. 그렇다고 답한 뒤 두 속성 가운데 하나를 `get_prediction`으로 보여 주고, 다른 하나도 따로 볼 수 있다고 알린다.
- **연도에 따른 변화는 예측하지 못한다.** 모델에 주는 입력에는 연도 정보가 없다. "내년에 어떻게 변해?"라는 질문에는 추세를 예측하는 모델이 아니라고 답한다. 이 모델이 하는 일은 조사를 돌리기 전에 그 문항의 응답 분포를 미리 그려 보는 것이다.
- **새 문항은 이 화면에서 예측하지 못한다.** 지금은 저장된 예측을 조회하는 버전이라 목록에 없는 문항은 답할 수 없다. 새 문항 예측은 모델을 직접 돌리는 실시간 버전에서 하며, 그 결과는 실측이 없어 검증되지 않은 실험적인 값이라고 알린다.
- **예측의 원인을 설명하지 않는다.** 모델은 분포만 내놓고 이유는 내놓지 않는다. "왜 이렇게 예측했나"에는 모델이 이유를 말해 주지 않는다고 답한다. 사회적 배경을 추측해 예측의 근거처럼 말하면 사후에 지어낸 설명이 된다.
- 연구원 보고서 같은 참고 문서 검색은 아직 연결되지 않았다. 물으면 그렇게 답한다.
- 모델이 잘 못 맞힌 결과도 그대로 말한다. 오차가 크면 크다고, 작년 실측을 그대로 쓴 것보다 못하면 못하다고 말한다.

# 속성과 집단
한 번에 속성 하나로만 나눈다.
{attrs}

# 어댑터
{adapters_text(adapters)}
집단 제외 모델에서 뺀 집단, 문항 제외 모델에서 뺀 문항을 설명할 때는 "보지 않고 예측한 오차"와 "보고 학습한 기본 모델의 오차"(`기본_모델_오차_읽기`)를 반드시 나란히 말한다. 이 비교가 그 화면에서 보여 주려는 것이다.

# {meta['predict_year']}년 문항 목록 (qkey | 종류 | 문구)
{catalog}"""


def adapters_text(lines: list[str]) -> str:
    return "\n".join(lines)


def _run_tool(service: Service, name: str, args: dict) -> tuple[dict, dict | None]:
    """도구를 돌린다. (대화 LLM에 줄 결과, 화면에 보낼 패널 이벤트)를 돌려준다."""
    if name == "search_questions":
        hits = service.questions(str(args.get("query", "")), limit=8)
        return {"결과": [{"qkey": h["qkey"], "종류": h["scale_type"], "문구": h["text"], "보기": h["options"]}
                         for h in hits]}, None
    if name == "get_prediction":
        qkey, attribute, adapter = args.get("qkey"), args.get("attribute"), args.get("adapter")
        brief = service.prediction_brief(qkey, attribute, adapter)
        state = {"qkey": qkey, "attribute": attribute, "adapter": adapter}
        return brief, {"state": state, "prediction": service.prediction(qkey, attribute, adapter)}
    raise NotFound(f"없는 도구: {name}")


def _user_content(text: str, panel: dict | None, service: Service) -> list[dict]:
    blocks = [{"type": "text", "text": text}]
    if panel and panel.get("qkey") in service.store.questions:
        q = service.store.questions[panel["qkey"]]
        blocks.append({"type": "text", "text":
                       f"[화면 상태] 지금 패널에 떠 있는 것 — 문항: {q.qkey} ({q.text}), "
                       f"속성: {panel.get('attribute')}, 어댑터: {panel.get('adapter')}"})
    return blocks


async def run_chat(client, service: Service, history: list[dict], panel: dict | None) -> AsyncIterator[tuple[str, dict]]:
    """한 질문을 처리하며 (이벤트 이름, 자료)를 내놓는다.

    이벤트: status(도구 진행), text(답변 조각), panel(패널에 띄울 예측), done, error.
    """
    import anthropic

    # 이전 대화는 글만 넘긴다. 마지막 사용자 메시지에만 화면 상태를 붙인다.
    messages = [{"role": m["role"], "content": m["content"]} for m in history[:-1] if m.get("content")]
    messages.append({"role": "user", "content": _user_content(history[-1]["content"], panel, service)})
    system = [{"type": "text", "text": system_prompt(service), "cache_control": {"type": "ephemeral"}}]
    tool_defs = tools(service)

    wrote_text = False   # 앞 차례에서 글을 썼으면 다음 차례의 글과 문단을 나눈다
    for _ in range(MAX_TURNS):
        need_break = wrote_text
        try:
            async with client.messages.stream(
                model=config.CHAT_MODEL,
                max_tokens=config.CHAT_MAX_TOKENS,
                system=system,
                tools=tool_defs,
                tool_choice={"type": "auto"},
                thinking={"type": "adaptive"},
                output_config={"effort": config.CHAT_EFFORT},
                messages=messages,
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        if need_break:
                            yield "text", {"delta": "\n\n"}
                            need_break = False
                        yield "text", {"delta": event.text}
                response = await stream.get_final_message()
        except anthropic.AuthenticationError:
            yield "error", {"code": "auth", "message": "API 키가 올바르지 않습니다."}
            return
        except anthropic.APIConnectionError:
            yield "error", {"code": "connection", "message": "Claude API에 연결하지 못했습니다. 패널은 그대로 쓸 수 있습니다."}
            return
        except anthropic.APIStatusError as e:
            yield "error", {"code": "api_error", "message": f"Claude API 오류 ({e.status_code})"}
            return

        if any(b.type == "text" and b.text.strip() for b in response.content):
            wrote_text = True
        if response.stop_reason == "refusal":
            yield "error", {"code": "refusal", "message": "이 질문에는 답하지 못했습니다. 질문을 바꿔 다시 물어 주세요."}
            return
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            yield "done", {"stop_reason": response.stop_reason}
            return
        if response.stop_reason == "max_tokens":
            yield "error", {"code": "max_tokens", "message": "답변이 길이 제한에 걸려 끊겼습니다."}
            return

        results = []
        for block in tool_uses:
            yield "status", {"tool": block.name,
                             "text": "문항을 찾는 중" if block.name == "search_questions" else "예측을 불러오는 중"}
            try:
                result, panel_event = _run_tool(service, block.name, block.input if isinstance(block.input, dict) else {})
            except (NotFound, KeyError) as e:
                results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                "content": f"실패: {e}"})
                continue
            if panel_event:
                yield "panel", panel_event
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False)})
        # 사고 블록을 포함해 응답 내용을 그대로 되돌려 준다
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})

    yield "error", {"code": "too_many_turns", "message": "도구 호출이 너무 길어져 멈췄습니다."}
