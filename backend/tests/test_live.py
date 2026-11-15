"""실시간 백엔드의 배선을 아주 작은 무작위 모델로 확인한다. 실제 Bllossom으로 저장된 예측과 같은 값이
나오는지는 GPU가 있는 PC에서 `python -m suwon_chat.live_check`로 본다 (PRD A6)."""
import pytest
import torch

from suwon_chat.live import BASE, LiveBackend, LiveError
from suwon_chat.prompt import build_prompt, with_refusal

from tiny import make_tiny


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, service):
    store = service.store
    texts = [q.text + "".join(q.all_options) for q in store.questions.values()]
    texts += list(store.steering.values()) + [g for gs in store.groups.values() for g in gs]
    texts.append("Answer the following question keeping in mind your previous answers. Question as a choice between,\n새 문항 보기 하나 둘 셋")
    return make_tiny(tmp_path_factory.mktemp("tiny"), texts)


@pytest.fixture(scope="module")
def live(service, tiny):
    base, adapters = tiny
    backend = LiveBackend(service.store, base_model=str(base), adapter_dirs=adapters, device="cpu", dtype="float32",
                          load_4bit=False, batch_size=3)
    backend.load()
    return backend


def reference(tiny, store, qkey, attribute, adapter):
    """백엔드를 거치지 않고 predict_local.py와 같은 순서로 직접 계산한 값."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base, adapters = tiny
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    if adapter != BASE:
        model = PeftModel.from_pretrained(model, str(adapters[adapter]))
    model.eval()
    q = store.questions[qkey]
    n = len(q.all_options)
    letters = [(tok.encode("Answer:" + c, add_special_tokens=False)[-1], tok.encode("Answer: " + c, add_special_tokens=False)[-1])
               for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    out = {}
    for g in store.groups[attribute]:
        prompt = build_prompt(store.steering[attribute], store.groups[attribute], g, q.text, q.all_options)
        ids = tok(prompt, return_tensors="pt")          # 한 개씩, 채움 없이
        with torch.no_grad():
            p = torch.softmax(model(**ids).logits[0, -1].float(), -1)
        probs = [p[a].item() + p[b].item() for a, b in letters[:n]]
        out[g] = [x / sum(probs) for x in probs]
    return out


@pytest.mark.parametrize("adapter", ["exp1", "exp2", "exp3"])
def test_live_prediction_equals_direct_computation(live, tiny, service, adapter):
    """묶어서 돌리고(왼쪽 채움) 어댑터를 갈아 끼운 결과가, 어댑터 하나만 올려 한 개씩 돌린 값과 같다."""
    qkey, attribute = "q189_W2025", "AGE"
    got = live.predict(qkey, attribute, adapter)
    want = reference(tiny, service.store, qkey, attribute, adapter)
    want_base = reference(tiny, service.store, qkey, attribute, BASE)
    for g in service.store.groups[attribute]:
        assert got[g]["lora"] == pytest.approx(want[g], abs=2e-4)
        assert got[g]["base"] == pytest.approx(want_base[g], abs=2e-4)
        assert len(got[g]["lora"]) == len(service.store.questions[qkey].all_options)
        assert sum(got[g]["lora"]) == pytest.approx(1, abs=1e-3)


def test_adapters_give_different_answers_and_switching_back_is_stable(live):
    a = live.predict("q51_1_W2025", "TENURE", "exp1")
    b = live.predict("q51_1_W2025", "TENURE", "exp2")
    c = live.predict("q51_1_W2025", "TENURE", "exp3")
    first = "자가"
    assert a[first]["lora"] != b[first]["lora"] != c[first]["lora"]
    assert a[first]["base"] == b[first]["base"] == c[first]["base"]       # 학습 전 모델은 어댑터와 무관하다
    assert a[first]["lora"] != a[first]["base"]
    live._cache.clear()                                                    # 다시 돌려도 같은 값
    again = live.predict("q51_1_W2025", "TENURE", "exp1")
    assert again[first]["lora"] == pytest.approx(a[first]["lora"], abs=1e-5)


def test_repeated_request_is_not_recomputed(live, monkeypatch):
    live.predict("q138_W2025", "SEX", "exp1")
    monkeypatch.setattr(live, "_forward", lambda *a, **k: pytest.fail("이미 계산한 것을 다시 돌렸다"))
    live.predict("q138_W2025", "SEX", "exp1")


def test_new_question(live, service):
    options = with_refusal(["하나", "둘", "셋"])
    got = live.predict_new("새 문항", options, "SEX", "exp1")
    assert list(got) == service.store.groups["SEX"]
    assert all(len(v["lora"]) == 4 and sum(v["lora"]) == pytest.approx(1, abs=1e-3) for v in got.values())
    assert live.predict_new("새 문항", options, "SEX", "exp1") is not None
    other = live.predict_new("새 문항", with_refusal(["하나", "둘"]), "SEX", "exp1")        # 보기가 다르면 다른 문항이다
    assert len(other["남"]["lora"]) == 3


def test_service_in_live_mode_marks_the_source(live, service):
    from suwon_chat.service import Service
    svc = Service(service.store, live)
    p = svc.prediction("q189_W2025", "TENURE", "exp2")
    assert p["source"] == "live" and svc.meta()["source"] == "live"
    assert all(g["reference"] and g["actual"] for g in p["groups"])        # 실측과 견주는 값은 그대로 나온다
    assert 0 <= p["summary"]["error"] <= 1


def test_missing_adapter_or_model_is_reported(service, tiny, tmp_path):
    base, adapters = tiny
    with pytest.raises(LiveError, match="어댑터 exp2"):
        LiveBackend(service.store, base_model=str(base), adapter_dirs={**adapters, "exp2": tmp_path / "none"},
                    device="cpu", dtype="float32", load_4bit=False).load()
    with pytest.raises(LiveError, match="모델을 올리지 못했다"):
        LiveBackend(service.store, base_model=str(tmp_path / "no-model"), adapter_dirs=adapters,
                    device="cpu", dtype="float32", load_4bit=False).load()
    with pytest.raises(LiveError, match="cuda"):
        LiveBackend(service.store, base_model=str(base), adapter_dirs=adapters, device="cpu", dtype="float32",
                    load_4bit=True).load()


def test_problems_are_reported_before_the_weights_are_read(service, tiny, monkeypatch):
    """장치·정밀도·메모리 문제는 16GB를 읽기 전에, 대처법과 함께 알린다."""
    import suwon_chat.live as live_module
    base, adapters = tiny
    make = lambda **kw: LiveBackend(service.store, **{"base_model": str(base), "adapter_dirs": adapters,
                                                      "device": "cpu", "dtype": "float32", "load_4bit": False, **kw})
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained",
                        lambda *a, **k: pytest.fail("검사 전에 가중치를 읽기 시작했다"))
    with pytest.raises(LiveError, match="cuda 장치를 쓸 수 없다"):
        make(device="cuda").load()                                  # 이 PC에는 GPU가 없다
    with pytest.raises(LiveError, match="LIVE_DTYPE"):
        make(dtype="bf16").load()
    with pytest.raises(LiveError, match="LIVE_DEVICE"):
        make(device="gpu").load()
    # 8B 모델인데 메모리가 모자라면: 얼마가 필요한지와 대처법을 말한다
    big = LiveBackend(service.store, base_model=live_module.config.DEFAULT_BASE_MODEL, adapter_dirs=adapters,
                      device="cpu", dtype="bfloat16", load_4bit=False)
    monkeypatch.setattr(live_module, "_free_gb", lambda device: 6.5)
    with pytest.raises(LiveError, match=r"약 17 GB가 필요한데 지금 쓸 수 있는 것은 6.5 GB.*LIVE_4BIT.*조회 버전"):
        big.load()
    assert big.revision == live_module.config.LIVE_BASE_REVISION       # 허깅페이스에서 받을 때는 판을 고정한다
    assert make().revision is None                                     # 로컬 폴더에는 판이 없다


def test_batched_run_equals_one_by_one(service, tiny):
    """왼쪽을 채워 묶어 돌린 값이 한 개씩 돌린 값과 같다 (위치 번호를 맞춰 넘기는지)."""
    base, adapters = tiny
    make = lambda n: LiveBackend(service.store, base_model=str(base), adapter_dirs=adapters, device="cpu",
                                 dtype="float32", load_4bit=False, batch_size=n)
    one, many = make(1), make(7)
    a = one.predict("q127_3_W2025", "EMPLOYMENT", "exp2")              # 집단 7개, 프롬프트 길이가 서로 다르다
    b = many.predict("q127_3_W2025", "EMPLOYMENT", "exp2")
    for g in a:
        assert b[g]["lora"] == pytest.approx(a[g]["lora"], abs=2e-5)
        assert b[g]["base"] == pytest.approx(a[g]["base"], abs=2e-5)


# ---- 새 문항: 서비스, 대화 도구, API -----------------------------------------------
from test_chat import FakeClient, collect  # noqa: E402


@pytest.fixture(scope="module")
def live_service(live, service):
    from suwon_chat.service import Service
    return Service(service.store, live)


def test_new_prediction_has_no_actuals_and_is_marked_experimental(live_service):
    p = live_service.new_prediction("새 문항", ["하나", "둘", "셋"], True, "AGE", "exp1")
    assert p["experimental"] is True and p["source"] == "live" and "실험적" in p["note"]
    assert p["question"]["qkey"] is None and p["question"]["scale_type"] == "순서형"
    assert p["question"]["options"] == ["하나", "둘", "셋"]            # 무응답은 화면에 내지 않는다
    assert p["summary"]["error"] is None and p["summary"]["error_readable"] is None
    for g in p["groups"]:
        assert g["actual"] is None and g["error"] is None
        assert len(g["predicted"]) == 3 and sum(g["predicted"]) == pytest.approx(1, abs=1e-6)
        assert 1 <= g["predicted_mean"] <= 3
    kinds = lambda options, ordered: live_service.new_prediction("새 문항", options, ordered, "SEX", "exp1")["question"]["scale_type"]
    assert kinds(["하나", "둘"], True) == "이분형" and kinds(["하나", "둘", "셋"], False) == "명목형"
    assert live_service.new_prediction("새 문항", ["하나", "둘", "셋"], False, "SEX", "exp1")["groups"][0]["predicted_mean"] is None


def test_off_scale_option_is_shown_but_left_out_of_the_mean(live_service):
    options = ["1(하나)", "2", "3(셋)", "둘 다 아님"]
    p = live_service.new_prediction("새 문항", options, True, "SEX", "exp1", ["둘 다 아님"])
    q = p["question"]
    assert q["scale_type"] == "순서형" and q["options"] == options
    assert q["ordinal"] == [1.0, 2.0, 3.0, -1.0] and q["option_values"] == [1.0, 2.0, 3.0, None]
    assert (q["scale_min"], q["scale_max"]) == (1.0, 3.0)
    g = p["groups"][0]
    inside = g["predicted"][:3]
    assert g["predicted_mean"] == pytest.approx(sum((i + 1) * v for i, v in enumerate(inside)) / sum(inside))
    zero_based = live_service.new_prediction("새 문항", ["0(하나)", "1", "2(셋)"], True, "SEX", "exp1")
    assert zero_based["question"]["option_values"] == [0.0, 1.0, 2.0]          # 보기에 적힌 숫자대로


def test_new_prediction_rejects_bad_input(live_service, service):
    from suwon_chat.service import Invalid, NotFound, Unsupported
    for args in (("새 문항", ["하나", "둘"], True, "NOPE", "exp1"), ("새 문항", ["하나", "둘"], True, "SEX", "exp9")):
        with pytest.raises(NotFound):
            live_service.new_prediction(*args)
    twelve = [f"보기{i}" for i in range(12)]
    for args in (("", ["하나", "둘"], True, "SEX", "exp1"), ("새 문항", ["하나"], True, "SEX", "exp1"),
                 ("새 문항", twelve, True, "SEX", "exp1"),                     # 학습한 문항보다 보기가 많다
                 ("새 문항", ["하나", "하나"], True, "SEX", "exp1"),
                 ("새 문항", ["하나", "둘"], True, "SEX", "exp1", ["셋"])):     # 보기에 없는 척도 밖 보기
        with pytest.raises(Invalid):
            live_service.new_prediction(*args)
    assert len(live_service.new_prediction("새 문항", twelve[:11], False, "SEX", "exp1")["question"]["options"]) == 11
    with pytest.raises(Unsupported):                               # 조회 버전에서는 안 된다
        service.new_prediction("새 문항", ["하나", "둘"], True, "SEX", "exp1")


def test_new_question_tool_exists_only_in_live_mode(live_service, service):
    from suwon_chat.chat import system_prompt, tools
    names = lambda s: [t["name"] for t in tools(s)]
    assert "predict_new_question" in names(live_service) and "predict_new_question" not in names(service)
    assert "실시간" in system_prompt(live_service) and "새 문항 예측은 실험적인 값이다" in system_prompt(live_service)
    assert "새 문항은 이 화면에서 예측하지 못한다" in system_prompt(service)


def test_new_question_tool_sends_an_experimental_card(live_service):
    args = {"question": "새 문항", "options": ["하나", "둘", "셋"], "ordered": True, "off_scale_options": [],
            "attribute": "SEX", "adapter": "exp1"}
    client = FakeClient([{"tools": [("predict_new_question", args)], "stop": "tool_use"}, {"text": ["실험적 예측입니다."]}])
    events = collect(client, live_service, [{"role": "user", "content": "새 문항을 넣으면?"}])
    assert [e for e, _ in events] == ["status", "new_prediction", "text", "done"]
    assert dict(events)["new_prediction"]["prediction"]["experimental"] is True
    import json
    brief = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
    assert "실험적" in brief["주의"] and "오차" not in json.dumps(brief["집단별"], ensure_ascii=False)


def test_new_question_api(live_service, service, report_index, monkeypatch):
    from fastapi.testclient import TestClient
    from suwon_chat import api
    body = {"question": "새 문항", "options": ["하나", "둘", "셋"], "ordered": True, "attribute": "AGE"}
    with TestClient(api.app) as c:
        api.app.state.reports = report_index
        api.app.state.service = service
        assert c.post("/api/predict_new", json=body).status_code == 409          # 조회 버전
        assert c.get("/api/meta").json()["can_predict_new"] is False
        api.app.state.service = live_service
        r = c.post("/api/predict_new", json=body)
        assert r.status_code == 200 and r.json()["experimental"] is True and len(r.json()["groups"]) == 6
        meta = c.get("/api/meta").json()
        assert meta["can_predict_new"] is True and meta["source"] == "live" and meta["live"]["device"] == "cpu"
        assert c.post("/api/predict_new", json={**body, "options": ["하나"]}).status_code == 422
        assert c.post("/api/predict_new", json={**body, "options": [f"보기{i}" for i in range(12)]}).status_code == 422
        assert c.post("/api/predict_new", json={**body, "off_scale_options": ["없는 보기"]}).status_code == 422
        assert c.post("/api/predict_new", json={**body, "attribute": "NOPE"}).status_code == 404


def test_check_sample_includes_held_out_groups_and_questions(service):
    """확인 스크립트의 표본에는 그 어댑터가 학습에서 뺀 집단의 속성과 뺀 문항이 반드시 들어간다."""
    from suwon_chat import config
    from suwon_chat.live_check import pick_pairs
    for seed in (1, 2, 3):
        exp2 = pick_pairs(service.store, "exp2", 24, seed)
        assert {a for a, _ in config.ADAPTERS["exp2"]["held_out_groups"]} <= {a for _, a in exp2}
        exp3 = pick_pairs(service.store, "exp3", 24, seed)
        assert set(config.ADAPTERS["exp3"]["held_out_questions"]) <= {q for q, _ in exp3}
        kinds = {service.store.questions[q].scale_type for q, _ in pick_pairs(service.store, "exp1", 24, seed)}
        assert kinds == {"순서형", "이분형", "명목형"}
        assert len(exp2) == len(set(exp2)) == 24


class StoredAsLive:
    """저장된 예측을 실시간 값인 것처럼 돌려주는 가짜 백엔드. 확인 스크립트의 판정을 시험한다."""

    def __init__(self, store, swap=None, base_from=None, **_):
        self.store, self.swap, self.base_from = store, swap or {}, base_from

    def load(self):
        pass

    def describe(self):
        return {"device": "fake"}

    def predict(self, qkey, attribute, adapter):
        source = self.swap.get(adapter, adapter)
        out = {}
        for g in self.store.groups[attribute]:
            row = self.store.predictions[(source, qkey, attribute, g)]
            base = self.store.predictions[(self.base_from or source, qkey, attribute, g)]["lora" if self.base_from else "base"]
            out[g] = {"lora": row["lora"], "base": base}
        return out


def run_check(monkeypatch, service, argv, **fake):
    from suwon_chat import live_check
    monkeypatch.setattr(live_check, "Store", lambda: service.store)
    monkeypatch.setattr(live_check, "LiveBackend", lambda store, **kw: StoredAsLive(store, **fake))
    monkeypatch.setattr("sys.argv", ["live_check", *argv])
    return live_check.main()


def test_check_passes_when_live_equals_stored(monkeypatch, service, capsys):
    assert run_check(monkeypatch, service, ["--pairs", "12"]) == 0
    out = capsys.readouterr().out
    assert "빠른 점검 통과" in out and "시연에 써도 된다" not in out         # 일부만 보고 다 됐다고 하지 않는다


def test_check_fails_when_an_adapter_is_not_switched(monkeypatch, service, capsys):
    """exp2를 끼웠다고 하고 exp1의 값이 나오면 잡아낸다. (실측과의 오차 평균만 보면 못 잡는다.)"""
    assert run_check(monkeypatch, service, ["--pairs", "24"], swap={"exp2": "exp1"}) == 1
    out = capsys.readouterr().out
    assert "실패  맞는 어댑터가 끼워졌다" in out


def test_check_fails_when_the_untrained_model_is_wrong(monkeypatch, service, capsys):
    """학습 전 모델 자리에 어댑터의 값이 나오면(어댑터가 꺼지지 않으면) 잡아낸다."""
    assert run_check(monkeypatch, service, ["--pairs", "12", "--adapter", "exp1"], base_from="exp1") == 1
    assert "실패  학습 전 모델 예측이 저장된 값과 같다" in capsys.readouterr().out


# ---- 양자화 버전: 파일이 없을 때의 안내 (실제 추론은 4.9GB 모델이 있어야 해서 live_check로 확인한다) ----
def test_quantized_backend_reports_missing_files(service, tmp_path):
    from suwon_chat.quantized import QuantizedBackend, _quant_name
    adapters = {"exp1": tmp_path / "exp1.gguf"}
    with pytest.raises(LiveError, match=r"(?s)양자화 모델 파일이 없다.*--download"):
        QuantizedBackend(service.store, model_path=tmp_path / "none.gguf", adapter_files=adapters).load()
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    with pytest.raises(LiveError, match=r"(?s)어댑터 exp1의 GGUF 파일이 없다.*convert_adapters_gguf"):
        QuantizedBackend(service.store, model_path=tmp_path / "model.gguf", adapter_files=adapters).load()
    adapters["exp1"].write_bytes(b"GGUF")
    with pytest.raises(LiveError, match="양자화 모델을 올리지 못했다"):          # 깨진 파일
        QuantizedBackend(service.store, model_path=tmp_path / "model.gguf", adapter_files=adapters).load()
    assert _quant_name("llama-3-Korean-Bllossom-8B-Q4_K_M.gguf") == "Q4_K_M 양자화"
    assert _quant_name("llama-3-Korean-Bllossom-8B.Q8_0.gguf") == "Q8_0 양자화"


def test_converted_adapters_hold_the_same_weights():
    """llama.cpp용으로 바꾼 어댑터가 원래 어댑터와 같은 값을 담고 있다. q 쪽 B 행렬만 행 순서가 바뀐다."""
    import gguf
    import numpy as np
    from safetensors import safe_open
    from suwon_chat import config
    # llama.cpp가 q·k 가중치에 쓰는 행 순서: 머리(32개)마다 앞 절반과 뒤 절반을 번갈아 놓는다.
    # 허깅페이스의 (머리 h, 절반 j, 자리 i) 행이 llama.cpp에서는 (머리 h, 자리 i, 절반 j) 행이 된다.
    n_head, half = 32, 64
    order = np.empty(4096, dtype=int)
    for h in range(n_head):
        for j in range(2):
            for i in range(half):
                order[h * 128 + i * 2 + j] = h * 128 + j * half + i
    for name in config.ADAPTERS:
        folder = config.ADAPTER_DIR / name
        reader = gguf.GGUFReader(str(folder / "adapter.gguf"))
        tensors = {t.name: t for t in reader.tensors}
        fields = reader.fields
        assert bytes(fields["general.type"].parts[-1]).decode() == "adapter"
        assert bytes(fields["adapter.type"].parts[-1]).decode() == "lora"
        assert float(fields["adapter.lora.alpha"].parts[-1][0]) == 32.0
        with safe_open(str(folder / "adapter_model.safetensors"), "np") as f:
            keys = list(f.keys())
            assert len(tensors) == len(keys) == 128
            for key in keys:
                layer, module, ab = key.split(".")[4], key.split(".")[6], key.split(".")[7][-1].lower()
                out = f"blk.{layer}.{'attn_q' if module == 'q_proj' else 'attn_v'}.weight.lora_{ab}"
                want = f.get_tensor(key)
                if module == "q_proj" and ab == "b":
                    want = want[order]
                assert np.array_equal(np.asarray(tensors[out].data).reshape(want.shape), want), key
