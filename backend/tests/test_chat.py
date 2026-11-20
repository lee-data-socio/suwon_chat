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


def collect(client, service, history, panel=None, reports=None, grounding="full", concepts=None):
    async def go():
        return [e async for e in run_chat(client, service, history, panel, reports, grounding, concepts)]
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
    assert panel["compare"] == "actual"
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
    state = sent[2]["content"][1]["text"]
    assert "qkey=q189_W2025" in state and "attribute=TENURE" in state
    assert "점유형태" in state and "전체 학습 어댑터" in state and "경제·사회적 위치" in state


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
    # 문항 목록은 주지 않는다. 문항은 search_questions로 찾는다 (로컬 모델은 목록 전체를 받아 판단하기 어렵다)
    named = [q.qkey for q in service.store.target_questions if q.qkey in prompt]
    assert len(named) <= 4          # 문항 제외 어댑터가 뺀 문항만 이름이 나온다
    assert "먼저 `search_questions`로 찾고" in prompt and "질문을 어떻게 이해했나" in prompt
    for phrase in ["교차 집단", "연도에 따른 변화", "새 문항", "모델의 이유는 지어내지 않는다", "집단 사이의 차이를 과장하지 않는다",
                   "표본이 적은 집단", "전체 학습 어댑터", "지난_조사_평균", "예측에 쓴 근거가 아니다", "수원서베이로 학습한 AI"]:
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


def test_history_is_cleaned_before_it_reaches_the_api():
    from suwon_chat.chat import HISTORY_MESSAGES, MESSAGE_CHARS, normalize_history
    history = [{"role": "assistant", "content": "안녕하세요. 무엇을 볼까요?"},     # 화면의 인사말
               {"role": "user", "content": "첫 질문"}, {"role": "assistant", "content": "   "},
               {"role": "assistant", "content": "답" * (MESSAGE_CHARS + 50)}, {"role": "user", "content": " 둘째 질문 "}]
    out = normalize_history(history)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]        # 인사말과 빈 메시지는 빠진다
    assert len(out[1]["content"]) == MESSAGE_CHARS and out[2]["content"] == "둘째 질문"

    long = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(101)]
    out = normalize_history(long)
    assert len(out) <= HISTORY_MESSAGES and out[0]["role"] == "user" and out[-1]["content"] == "m100"


def test_greeting_first_history_is_sent_starting_with_user(service):
    client = FakeClient([{"text": ["네."]}])
    collect(client, service, [{"role": "assistant", "content": "안녕하세요"}, {"role": "user", "content": "질문"}])
    assert [m["role"] for m in client.calls[0]["messages"]] == ["user"]


def test_blank_question_is_rejected_without_calling_the_api(service):
    client = FakeClient([])
    events = collect(client, service, [{"role": "user", "content": "   "}])
    assert [e for e, _ in events] == ["error"] and client.calls == []


def test_unexpected_tool_failure_does_not_break_the_stream(service, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("예상하지 못한 실패")
    monkeypatch.setattr(service, "prediction_brief", boom)
    args = {"qkey": "q189_W2025", "attribute": "TENURE", "adapter": "exp1"}
    client = FakeClient([{"tools": [("get_prediction", args)], "stop": "tool_use"}, {"text": ["지금은 가져올 수 없습니다."]}])
    events = collect(client, service, [{"role": "user", "content": "보여 줘"}])
    assert [e for e, _ in events] == ["status", "text", "done"]
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_truncated_text_answer_says_so(service):
    client = FakeClient([{"text": ["길게 쓰다가"], "stop": "max_tokens"}])
    events = collect(client, service, [{"role": "user", "content": "..."}])
    text = "".join(d["delta"] for e, d in events if e == "text")
    assert text.startswith("길게 쓰다가") and "끊겼습니다" in text
    assert events[-1] == ("done", {"stop_reason": "max_tokens"})


def test_question_key_is_restricted_to_the_catalogue(service):
    props = {t["name"]: t for t in tools(service)}["get_prediction"]["input_schema"]["properties"]
    assert len(props["qkey"]["enum"]) == 113 and "q189_W2025" in props["qkey"]["enum"]


def test_consecutive_messages_from_the_same_side_are_merged():
    from suwon_chat.chat import normalize_history
    # 첫 질문의 답변이 오류로 끝나 비어 있으면 사용자 메시지가 연달아 온다
    out = normalize_history([{"role": "user", "content": "첫 질문"}, {"role": "assistant", "content": ""},
                             {"role": "user", "content": "둘째 질문"}])
    assert out == [{"role": "user", "content": "첫 질문\n\n둘째 질문"}]


def _compare_for(service, args):
    client = FakeClient([{"tools": [("get_prediction", args)], "stop": "tool_use"}, {"text": ["네."]}])
    return dict(collect(client, service, [{"role": "user", "content": "보여 줘"}]))["panel"]["compare"]


def test_compare_choice_is_passed_to_the_panel_when_it_makes_sense(service):
    base = {"qkey": "q132_1_W2025", "attribute": "TENURE"}
    assert _compare_for(service, {**base, "adapter": "exp1", "compare": "last_year"}) == "last_year"
    assert _compare_for(service, {**base, "adapter": "exp1", "compare": "zero_shot"}) == "zero_shot"
    assert _compare_for(service, {**base, "adapter": "exp2", "compare": "reference"}) == "reference"
    # 전체 학습 어댑터를 자기 자신과 견줄 수는 없다
    assert _compare_for(service, {**base, "adapter": "exp1", "compare": "reference"}) == "actual"
    # 고르지 않으면 빼고 학습한 어댑터는 전체 학습 어댑터와 견준다
    assert _compare_for(service, {**base, "adapter": "exp3"}) == "reference"
    # 작년에 같은 문항이 없으면 작년 비교를 켜지 않는다
    no_history = {"qkey": "q74_W2025", "attribute": "SEX", "adapter": "exp1", "compare": "last_year"}
    assert _compare_for(service, no_history) == "actual"


def test_question_keys_are_removed_from_the_answer_text(service):
    """문항 키가 답변에 새어 나오면 글이 나가기 전에 지운다. 조각 사이에 걸쳐 와도 지운다."""
    from suwon_chat.chat import KeyFilter
    def run(chunks):
        f = KeyFilter()
        return "".join(f.feed(c) for c in chunks) + f.flush()
    assert run(["일자리 지원 문항(q132_1)을 점유형태별로 봤습니다."]) == "일자리 지원 문항을 점유형태별로 봤습니다."
    assert run(["자부심 문항(q1", "38_W2025)을 보면"]) == "자부심 문항을 보면"
    assert run(["문항 ", "q189", " 은 0~10점입니다. 평균 5.47점, 7점 척도"]) == "문항 은 0~10점입니다. 평균 5.47점, 7점 척도"
    assert run(["오차는 0.36점(WD 0.036)입니다.", " 2024년 5.8점"]) == "오차는 0.36점(WD 0.036)입니다. 2024년 5.8점"      # 숫자와 괄호는 건드리지 않는다
    assert run(["Q4_K_M, exp1, quality"]) == "Q4_K_M, exp1, quality"
    client = FakeClient([{"text": ["경제·사회적 위치 문항(q189", "_W2025)을 봤습니다."]}])
    events = collect(client, service, [{"role": "user", "content": "보여 줘"}])
    assert "".join(d["delta"] for e, d in events if e == "text") == "경제·사회적 위치 문항을 봤습니다."


# ---- 근거 단계: 같은 대화 모델에 근거를 얼마나 주느냐만 바꾼다 ----------------------------
PANEL = {"qkey": "q189_W2025", "attribute": "TENURE", "adapter": "exp1"}


def test_general_level_gets_only_the_question(service, report_index):
    client = FakeClient([{"text": ["일반적으로 자가 거주자가 높게 답하는 경향이 있습니다."]}])
    history = [{"role": "user", "content": "점유형태별로 경제·사회적 위치를 어떻게 답할지 예측해 줘"}]
    events = collect(client, service, history, PANEL, report_index, grounding="none")
    assert [e for e, _ in events] == ["text", "done"]
    call = client.calls[0]
    assert "tools" not in call and "tool_choice" not in call
    assert call["system"] == [{"type": "text", "text": chat.GENERAL_PROMPT}]
    # 화면 상태(문항 문구, qkey)도, 문항 목록도 넘어가지 않는다
    assert call["messages"] == history
    assert "q189" not in json.dumps(call, ensure_ascii=False, default=str)


def test_data_level_has_predictions_but_no_documents(service, report_index):
    args = {**PANEL, "compare": "actual"}
    client = FakeClient([{"tools": [("get_prediction", args)], "stop": "tool_use"}, {"text": ["네."]}])
    events = collect(client, service, [{"role": "user", "content": "보여 줘"}], None, report_index, grounding="data")
    assert [e for e, _ in events] == ["status", "panel", "text", "done"]          # context 이벤트가 없다
    first = client.calls[0]
    assert "search_reports" not in [t["name"] for t in first["tools"]]
    assert chat.REPORTS_WITHHELD in first["system"][0]["text"]
    brief = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
    assert "참고_문서" not in brief


def test_full_level_attaches_documents(service, report_index):
    args = {**PANEL, "compare": "actual"}
    client = FakeClient([{"tools": [("get_prediction", args)], "stop": "tool_use"}, {"text": ["네."]}])
    events = collect(client, service, [{"role": "user", "content": "보여 줘"}], None, report_index)
    assert "context" in [e for e, _ in events]
    assert "search_reports" in [t["name"] for t in client.calls[0]["tools"]]
    assert "참고_문서" in json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])


def test_unknown_level_is_rejected_without_calling_the_api(service):
    client = FakeClient([])
    events = collect(client, service, [{"role": "user", "content": "질문"}], grounding="anything")
    assert [e for e, _ in events] == ["error"] and client.calls == []



# ---- 문항 찾기: full은 개념 사전, data는 단어 검색 ------------------------------------------
def _search_result(service, report_index, concept_index, grounding):
    client = FakeClient([{"tools": [("search_questions", {"query": "월세 사는 사람들 형편 살림살이"})], "stop": "tool_use"},
                         {"text": ["네."]}])
    collect(client, service, [{"role": "user", "content": "월세 사는 사람들 형편 어때?"}], None, report_index,
            grounding, concept_index)
    tool = next(t for t in client.calls[0]["tools"] if t["name"] == "search_questions")
    return tool, json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])


def test_full_level_finds_questions_with_the_concept_dictionary(service, report_index, concept_index):
    tool, result = _search_result(service, report_index, concept_index, "full")
    assert "개념 사전" in tool["description"]
    assert all("개념" in r and "정의" in r for r in result["결과"])
    assert {"q189_W2025", "q121_2_W2025"} & {r["qkey"] for r in result["결과"]}     # 형편·살림살이를 일상 표현으로 둔 문항


def test_data_level_finds_questions_by_words_only(service, report_index, concept_index):
    tool, result = _search_result(service, report_index, concept_index, "data")
    assert "글자 그대로" in tool["description"]
    assert all("개념" not in r for r in result["결과"])


def test_understanding_shows_both_searches_for_the_users_own_words(service, report_index, concept_index):
    q = "가구 부채 빚 대출"
    client = FakeClient([{"tools": [("search_questions", {"query": "빚"})], "stop": "tool_use"},
                         {"tools": [("search_questions", {"query": "대출"})], "stop": "tool_use"},
                         {"text": ["네."]}])
    events = collect(client, service, [{"role": "user", "content": q}], None, report_index, "data", concept_index)
    shown = [d for e, d in events if e == "understanding"]
    assert len(shown) == 1                                   # 한 답변에 한 번
    u = shown[0]
    assert u["query"] == q                                   # 대화 모델의 검색어가 아니라 사용자의 질문 그대로
    assert u["concepts_available"] and u["concepts"][0]["qkey"] == "q28_W2025" and u["concepts"][0]["concept"]
    assert all(w["concept"] is None for w in u["words"])


def test_no_understanding_when_no_question_search(service):
    args = {**PANEL, "compare": "actual"}
    client = FakeClient([{"tools": [("get_prediction", args)], "stop": "tool_use"}, {"text": ["네."]}])
    events = collect(client, service, [{"role": "user", "content": "연령별로 바꿔 줘"}])
    assert "understanding" not in [e for e, _ in events]
