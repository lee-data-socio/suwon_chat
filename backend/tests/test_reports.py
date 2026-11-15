import json

import numpy as np
import pytest

from suwon_chat.reports import ReportError, ReportIndex, build_index, load_documents, parse_document, plain

from conftest import fake_embed
from test_chat import FakeClient, collect


def test_document_is_split_into_pages(tmp_path):
    p = tmp_path / "a.md"
    p.write_text('---\nid: R09\ntitle: "[시연용 가상 문서] 제목"\nsummary: "요약"\ntopics: ["가", "나"]\n---\n'
                 "<!-- page 1 -->\n# 제목\n\n첫 쪽.\n\n<!-- page 2 -->\n## 둘째\n\n둘째 쪽.\n", encoding="utf-8")
    d = parse_document(p)
    assert (d.doc_id, d.title, d.summary, d.topics) == ("R09", "[시연용 가상 문서] 제목", "요약", ["가", "나"])
    assert d.synthetic is True
    assert d.pages == [{"page": 1, "text": "# 제목\n\n첫 쪽."}, {"page": 2, "text": "## 둘째\n\n둘째 쪽."}]
    assert plain(d.pages[0]["text"]) == "제목 첫 쪽."


@pytest.mark.parametrize("text", ["본문만 있다", "---\nid: R1\n---\n쪽 표시가 없다", "---\ntitle: x\n---\n<!-- page 1 -->\n글"])
def test_malformed_document_is_rejected(tmp_path, text):
    p = tmp_path / "bad.md"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ReportError):
        parse_document(p)


def test_duplicate_ids_are_rejected(tmp_path):
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text("---\nid: R1\ntitle: t\n---\n<!-- page 1 -->\n글", encoding="utf-8")
    with pytest.raises(ReportError):
        load_documents(tmp_path)


def test_question_gets_the_closest_pages(report_index):
    items = report_index.for_question("q51_1_W2025")          # 버스 인프라 문항
    assert items[0]["doc_id"] == "R01" and items[0]["synthetic"] is True
    assert items[0]["title"].startswith("[시연용 가상 문서]") and items[0]["pages"] == 2
    assert report_index.for_question("q189_W2025")[0]["doc_id"] == "R02"     # 경제·사회적 위치 문항
    assert report_index.for_question("q183_W2025")[0]["doc_id"] == "R03"     # 반려동물 문항
    assert report_index.for_question("q183_W2025")[0]["synthetic"] is False  # 표시가 없는 제목은 실제 문서로 본다
    scores = [it["score"] for it in items]
    assert scores == sorted(scores, reverse=True)


def test_free_text_search(report_index):
    hits = report_index.search("반려동물이 있는 가구")
    assert hits[0]["doc_id"] == "R03" and hits[0]["page"] == 1
    assert report_index.search("   ") == []


def test_at_most_two_pages_per_document(tmp_path):
    for i in range(1, 3):
        pages = "".join(f"<!-- page {n} -->\n버스 인프라 이야기 {n}\n" for n in range(1, 5))
        (tmp_path / f"d{i}.md").write_text(f"---\nid: D{i}\ntitle: 문서 {i}\n---\n{pages}", encoding="utf-8")
    build_index(tmp_path, {"q": "버스 인프라"}, fake_embed, tmp_path / "i.json", tmp_path / "v.npy")
    idx = ReportIndex(tmp_path / "i.json", tmp_path / "v.npy", fake_embed, min_score=0.0, margin=1.0)
    docs = [it["doc_id"] for it in idx.for_question("q")]
    assert len(docs) == 3 and max(docs.count(d) for d in set(docs)) == 2


def test_unrelated_pages_are_not_shown(report_index):
    """관련 문서가 없으면 아무것도 띄우지 않는다. 1등과 많이 벌어진 대목도 버린다."""
    loose = report_index.for_question("q51_1_W2025")
    top = loose[0]["score"]
    strict = ReportIndex.__new__(ReportIndex)
    strict.__dict__.update(report_index.__dict__)
    strict.min_score, strict.margin = top + 0.01, 1.0
    assert strict.for_question("q51_1_W2025") == [] and strict.search("버스 인프라") == []      # 1등도 문턱 아래
    strict.min_score, strict.margin = 0.0, 0.0
    assert [it["score"] for it in strict.for_question("q51_1_W2025")] == [top]                # 1등만 남는다


def test_missing_index_means_unavailable(tmp_path):
    idx = ReportIndex(tmp_path / "none.json", tmp_path / "none.npy", fake_embed)
    assert idx.available is False and idx.for_question("q189_W2025") == [] and idx.search("버스") == []


def test_search_is_empty_when_the_embedding_model_fails(report_index):
    broken = ReportIndex.__new__(ReportIndex)
    broken.__dict__.update(report_index.__dict__)
    broken.embed = lambda texts, kind: np.zeros((len(texts), 1), np.float32)   # api._guarded가 실패 때 내놓는 모양
    assert broken.search("버스") == []
    assert broken.for_question("q51_1_W2025")          # 미리 골라 둔 것은 모델 없이도 나온다


# ---- 대화와 API ----------------------------------------------------------------
def test_prediction_is_followed_by_context_for_that_question(service, report_index):
    args = {"qkey": "q51_1_W2025", "attribute": "TENURE", "adapter": "exp1", "compare": "actual"}
    client = FakeClient([{"tools": [("get_prediction", args)], "stop": "tool_use"}, {"text": ["답."]}])
    events = collect(client, service, [{"role": "user", "content": "버스"}], reports=report_index)
    assert [e for e, _ in events] == ["status", "panel", "context", "text", "done"]
    ctx = dict(events)["context"]
    assert ctx["qkey"] == "q51_1_W2025" and ctx["items"][0]["doc_id"] == "R01"
    # 참고 문서는 화면으로만 간다. 대화 모델이 받은 도구 결과에는 없다.
    result = client.calls[1]["messages"][-1]["content"][0]["content"]
    assert "가상 문서" not in result and "R01" not in result


def test_search_reports_tool_returns_passages_and_shows_them(service, report_index):
    client = FakeClient([{"tools": [("search_reports", {"query": "주관적 계층 인식"})], "stop": "tool_use"},
                         {"text": ["관련 문서가 있습니다."]}])
    events = collect(client, service, [{"role": "user", "content": "관련 보고서 있어?"}], reports=report_index)
    assert [e for e, _ in events] == ["status", "context", "text", "done"]
    assert dict(events)["status"]["text"] == "참고 문서를 찾는 중"
    result = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
    first = result["결과"][0]
    assert first["문서_제목"] == "[시연용 가상 문서] 주관적 계층 인식" and first["쪽"] == 1 and first["가상_문서"] is True
    assert "예측에 쓰이지 않은" in result["안내"]


def test_report_tool_and_rules_exist_only_when_reports_are_connected(service, report_index, tmp_path):
    from suwon_chat.chat import system_prompt, tools
    names = lambda r: [t["name"] for t in tools(service, r)]
    assert "search_reports" in names(report_index)
    assert "참고 문서는 예측의 근거가 아니다" in system_prompt(service, report_index)
    off = ReportIndex(tmp_path / "x.json", tmp_path / "x.npy")
    assert "search_reports" not in names(off) and "search_reports" not in names(None)
    assert "아직 연결되지 않았다" in system_prompt(service, off)


def test_context_and_report_endpoints(service, report_index):
    from fastapi.testclient import TestClient
    from suwon_chat import api
    with TestClient(api.app) as c:
        api.app.state.service, api.app.state.reports = service, report_index
        assert c.get("/api/meta").json()["reports"] == {"available": True, "count": 3}
        ctx = c.get("/api/context", params={"qkey": "q189_W2025"}).json()
        assert ctx["available"] is True and ctx["items"][0]["doc_id"] == "R02" and "경제·사회적 위치" in ctx["query"]
        assert c.get("/api/context", params={"qkey": "q189_W2024"}).status_code == 404
        assert [d["doc_id"] for d in c.get("/api/reports").json()] == ["R01", "R02", "R03"]
        doc = c.get("/api/reports/R01").json()
        assert [p["page"] for p in doc["pages"]] == [1, 2] and doc["synthetic"] is True
        assert c.get("/api/reports/R99").status_code == 404
