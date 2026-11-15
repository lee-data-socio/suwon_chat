import pytest
from fastapi.testclient import TestClient

from suwon_chat import api


@pytest.fixture()
def client(service, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with TestClient(api.app) as c:
        api.app.state.service = service      # 테스트용으로 새로 색인한 DB를 쓴다
        yield c


def test_meta(client):
    meta = client.get("/api/meta").json()
    assert meta["source"] == "precomputed"
    assert [a["id"] for a in meta["adapters"]] == ["exp1", "exp2", "exp3"]
    assert sum(len(a["groups"]) for a in meta["attributes"]) == 33
    assert all(g["n"] > 0 for a in meta["attributes"] for g in a["groups"])


def test_prediction_and_not_found(client):
    ok = client.get("/api/prediction", params={"qkey": "q189_W2025", "attribute": "TENURE", "adapter": "exp1"})
    assert ok.status_code == 200 and len(ok.json()["groups"]) == 3
    assert client.get("/api/prediction", params={"qkey": "nope", "attribute": "TENURE"}).status_code == 404
    assert client.get("/api/prediction", params={"qkey": "q189_W2025"}).status_code == 422


def test_questions_and_scenarios(client):
    assert len(client.get("/api/questions").json()) == 113
    assert client.get("/api/questions", params={"q": "버스 인프라"}).json()[0]["qkey"] == "q51_1_W2025"
    assert len(client.get("/api/scenarios").json()) >= 5


def test_chat_without_key_reports_it_as_an_event(client):
    assert client.get("/api/chat/status").json()["available"] is False
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "안녕"}]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert "event: error" in r.text and "no_api_key" in r.text


def test_chat_rejects_malformed_history(client):
    assert client.post("/api/chat", json={"messages": []}).status_code == 422
    assert client.post("/api/chat", json={"messages": [{"role": "assistant", "content": "x"}]}).status_code == 400
