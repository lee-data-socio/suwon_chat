"""대화 루프를 가짜 Claude 클라이언트로 돌려 본다. 실제 API는 부르지 않는다."""
import asyncio
import json
from types import SimpleNamespace

from suwon_chat import chat
from suwon_chat.chat import run_chat, system_prompt, tools


class FakeStream:
    def __init__(self, turn):
        self.turn = turn

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for text in self.turn.get("text", []):
                yield SimpleNamespace(type="text", text=text)
        return gen()

    async def get_final_message(self):
        content = [SimpleNamespace(type="thinking", thinking="", signature="sig")]
        content += [SimpleNamespace(type="text", text=t) for t in self.turn.get("text", [])]
        content += [SimpleNamespace(type="tool_use", id=f"tu_{i}", name=name, input=args)
                    for i, (name, args) in enumerate(self.turn.get("tools", []))]
        return SimpleNamespace(content=content, stop_reason=self.turn.get("stop", "end_turn"))


class FakeClient:
    """미리 정한 차례대로 응답한다. 받은 요청은 `calls`에 남긴다."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return FakeStream(self.turns.pop(0))


def collect(client, service, history, panel=None):
    async def go():
        return [e async for e in run_chat(client, service, history, panel)]
    return asyncio.run(go())


def test_prediction_goes_to_panel_and_brief_goes_to_model(service):
    args = {"qkey": "q189_W2025", "attribute": "TENURE", "adapter": "exp1"}
    client = FakeClient([
        {"tools": [("get_prediction", args)], "stop": "tool_use"},
        {"text": ["자가 집단이 ", "가장 높습니다."]},
    ])
    events = collect(client, service, [{"role": "user", "content": "점유형태별로 보여 줘"}])
    assert [e for e, _ in events] == ["status", "panel", "text", "text", "done"]

    panel = dict(events)["panel"]
    assert panel["state"] == args
    assert panel["prediction"] == service.prediction(**args)        # 차트 자료는 서비스가 만든 그대로 간다

    # 두 번째 요청: 모델에는 요약이 도구 결과로 돌아가고, 사고 블록이 든 응답이 그대로 붙는다
    second = client.calls[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["content"][0].type == "thinking"
    result = second[-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "tu_0"
    assert json.loads(result["content"]) == service.prediction_brief(**args)


def test_request_uses_configured_model_and_auto_tool_choice(service, monkeypatch):
    monkeypatch.setattr(chat.config, "CHAT_MODEL", "claude-test-model")
    monkeypatch.setattr(chat.config, "CHAT_EFFORT", "medium")
    client = FakeClient([{"text": ["네."]}])
    collect(client, service, [{"role": "user", "content": "안녕"}])
    call = client.calls[0]
    assert call["model"] == "claude-test-model"
    assert call["output_config"] == {"effort": "medium"}
    assert call["tool_choice"] == {"type": "auto"}
    assert call["thinking"] == {"type": "adaptive"}
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in call["tools"])


def test_panel_state_is_attached_to_last_user_message_only(service):
    client = FakeClient([{"text": ["네."]}])
    history = [{"role": "user", "content": "첫 질문"}, {"role": "assistant", "content": "첫 답"},
               {"role": "user", "content": "속성을 연령으로 바꿔 줘"}]
    collect(client, service, history, {"qkey": "q189_W2025", "attribute": "TENURE", "adapter": "exp1"})
    sent = client.calls[0]["messages"]
    assert sent[0] == {"role": "user", "content": "첫 질문"}
    assert sent[1] == {"role": "assistant", "content": "첫 답"}
    assert sent[2]["content"][0]["text"] == "속성을 연령으로 바꿔 줘"
    assert "q189_W2025" in sent[2]["content"][1]["text"] and "TENURE" in sent[2]["content"][1]["text"]


def test_bad_tool_input_is_returned_as_error_and_loop_continues(service):
    client = FakeClient([
        {"tools": [("get_prediction", {"qkey": "q999_W2025", "attribute": "AGE", "adapter": "exp1"})], "stop": "tool_use"},
        {"text": ["그 문항은 없습니다."]},
    ])
    events = collect(client, service, [{"role": "user", "content": "q999 보여 줘"}])
    assert [e for e, _ in events] == ["status", "text", "done"]         # 패널은 바뀌지 않는다
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True


def test_search_tool_returns_candidates_without_touching_panel(service):
    client = FakeClient([
        {"tools": [("search_questions", {"query": "버스 인프라"})], "stop": "tool_use"},
        {"text": ["찾았습니다."]},
    ])
    events = collect(client, service, [{"role": "user", "content": "버스 문항 찾아 줘"}])
    assert "panel" not in [e for e, _ in events]
    result = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
    assert result["결과"][0]["qkey"] == "q51_1_W2025"


def test_refusal_is_reported_and_no_tool_runs(service):
    client = FakeClient([{"tools": [("get_prediction", {})], "stop": "refusal"}])
    events = collect(client, service, [{"role": "user", "content": "..."}])
    assert [e for e, _ in events] == ["error"]
    assert events[0][1]["code"] == "refusal"


def test_truncated_tool_call_is_not_executed(service):
    args = {"qkey": "q189_W2025", "attribute": "TENURE", "adapter": "exp1"}
    client = FakeClient([{"tools": [("get_prediction", args)], "stop": "max_tokens"}])
    events = collect(client, service, [{"role": "user", "content": "..."}])
    assert [e for e, _ in events] == ["error"] and events[0][1]["code"] == "max_tokens"


def test_loop_stops_after_too_many_tool_turns(service):
    turn = {"tools": [("search_questions", {"query": "버스"})], "stop": "tool_use"}
    client = FakeClient([turn] * (chat.MAX_TURNS + 1))
    events = collect(client, service, [{"role": "user", "content": "..."}])
    assert events[-1][0] == "error" and events[-1][1]["code"] == "too_many_turns"
    assert len(client.calls) == chat.MAX_TURNS


def test_system_prompt_lists_every_question_and_the_limits(service):
    prompt = system_prompt(service)
    for q in service.store.target_questions:
        assert q.qkey in prompt
    for phrase in ["교차 집단", "연도에 따른 변화", "새 문항", "예측의 원인", "집단 사이의 차이를 과장하지 않는다"]:
        assert phrase in prompt


def test_tool_enums_cover_all_attributes_and_adapters(service):
    props = {t["name"]: t for t in tools(service)}["get_prediction"]["input_schema"]["properties"]
    assert set(props["attribute"]["enum"]) == set(service.store.groups)
    assert props["adapter"]["enum"] == ["exp1", "exp2", "exp3"]


def test_text_before_and_after_a_tool_call_is_separated_by_a_paragraph_break(service):
    args = {"qkey": "q189_W2025", "attribute": "SEX", "adapter": "exp1"}
    client = FakeClient([
        {"text": ["교차 집단은 예측하지 못합니다."], "tools": [("get_prediction", args)], "stop": "tool_use"},
        {"text": ["성별로 ", "보면 이렇습니다."]},
    ])
    events = collect(client, service, [{"role": "user", "content": "30대 여성은?"}])
    text = "".join(d["delta"] for e, d in events if e == "text")
    assert text == "교차 집단은 예측하지 못합니다.\n\n성별로 보면 이렇습니다."
