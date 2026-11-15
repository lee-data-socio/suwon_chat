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
    items = report_index.for_question("q51_1_W2025")          # 버스 인프라 문항. R01이 q51_을 다룬다고 적혀 있다
    assert [it["doc_id"] for it in items] == ["R01", "R01"] and items[0]["synthetic"] is True
    assert items[0]["title"].startswith("[시연용 가상 문서]") and items[0]["pages"] == 2
    assert items[0]["summary"] == "버스 인프라에 대한 인식을 다룬다."
    scores = [it["score"] for it in items]
    assert scores == sorted(scores, reverse=True)
    # 어느 문서에도 적혀 있지 않은 문항은 임베딩으로 가장 가까운 문서 하나만
    assert [it["doc_id"] for it in report_index.for_question("q189_W2025")] == ["R02"]     # 경제·사회적 위치
    pet = report_index.for_question("q183_W2025")                                          # 반려동물
    assert [it["doc_id"] for it in pet] == ["R03"] and pet[0]["synthetic"] is False        # 표시가 없는 제목은 실제 문서로 본다


def test_listed_document_wins_over_a_closer_unlisted_one(tmp_path):
    """문항을 다룬다고 적힌 문서가 있으면, 글이 더 닮은 다른 문서가 있어도 그 문서만 띄운다."""
    (tmp_path / "a.md").write_text('---\nid: A\ntitle: 환경 만족도\nquestions: ["q137_"]\n---\n<!-- page 1 -->\n생활환경 이야기', encoding="utf-8")
    (tmp_path / "b.md").write_text("---\nid: B\ntitle: 버스 인프라\n---\n<!-- page 1 -->\n버스 인프라가 잘 구축되어 있다", encoding="utf-8")
    questions = {"q137_2_W2025": "버스 인프라가 잘 구축되어 있다", "q137_W2025": "버스 인프라", "q51_1_W2025": "버스 인프라가 잘 구축되어 있다"}
    build_index(tmp_path, questions, fake_embed, tmp_path / "i.json", tmp_path / "v.npy")
    idx = ReportIndex(tmp_path / "i.json", tmp_path / "v.npy", fake_embed, unlisted_min_score=0.5)
    assert [it["doc_id"] for it in idx.for_question("q137_2_W2025")] == ["A"]     # 적힌 문서
    assert [it["doc_id"] for it in idx.for_question("q137_W2025")] == ["B"]       # "q137_"는 q137 자체와는 다르다
    assert [it["doc_id"] for it in idx.for_question("q51_1_W2025")] == ["B"]      # 적히지 않은 문항은 임베딩으로
    strict = ReportIndex(tmp_path / "i.json", tmp_path / "v.npy", fake_embed, unlisted_min_score=1.01)
    assert strict.for_question("q51_1_W2025") == []                               # 문턱 아래면 띄우지 않는다
    assert [it["doc_id"] for it in strict.for_question("q137_2_W2025")] == ["A"]  # 적힌 문서는 문턱과 무관하다


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
    docs = [it["doc_id"] for it in idx.search("버스 인프라")]
    assert len(docs) == 3 and max(docs.count(d) for d in set(docs)) == 2


def test_free_search_drops_unrelated_pages(report_index):
    """자유 검색: 1등이 문턱 아래면 아무것도 돌려주지 않고, 1등과 많이 벌어진 대목은 버린다."""
    top = report_index.search("버스 인프라")[0]["score"]
    strict = ReportIndex.__new__(ReportIndex)
    strict.__dict__.update(report_index.__dict__)
    strict.min_score, strict.margin = top + 0.01, 1.0
    assert strict.search("버스 인프라") == []
    strict.min_score, strict.margin = 0.0, 0.0
    assert [it["score"] for it in strict.search("버스 인프라")] == [top]


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
        assert c.get("/api/reports/R01").json()["questions"] == ["q51_"]
        assert c.get("/api/context", params={"qkey": "q189_W2024"}).status_code == 404
        assert [d["doc_id"] for d in c.get("/api/reports").json()] == ["R01", "R02", "R03"]
        doc = c.get("/api/reports/R01").json()
        assert [p["page"] for p in doc["pages"]] == [1, 2] and doc["synthetic"] is True
        assert c.get("/api/reports/R99").status_code == 404
