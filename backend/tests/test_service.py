"""서비스가 내놓는 숫자를 정리본 결과표(01_결과표)의 값과 견준다."""
from collections import defaultdict

import pytest

from suwon_chat import config
from suwon_chat.scenarios import SCENARIOS
from suwon_chat.service import NotFound

from conftest import result_rows

BLLOSSOM = "bllossom-8b-b128-lr5e-5"


def _cells(service, adapter):
    """(문항, 속성, 집단) → 그 칸의 서비스 응답."""
    out = {}
    for q in service.store.target_questions:
        for attr in service.store.groups:
            for g in service.prediction(q.qkey, attr, adapter)["groups"]:
                out[(q.qkey, attr, g["group"])] = g
    return out


@pytest.fixture(scope="module")
def cells(service):
    return {adapter: _cells(service, adapter) for adapter in config.ADAPTERS}


def test_self_check_matches_report(service):
    assert service.store.self_check == {
        "actual_rows": 14850,
        "prediction_rows": {"exp1": 3729, "exp2": 3729, "exp3": 3729},
        "scored_cells": 1485,
        "exp1_wd": 0.0413,
    }


def test_every_cell_is_served(cells):
    for adapter in config.ADAPTERS:
        assert len(cells[adapter]) == 3729


@pytest.mark.parametrize("experiment,adapter,n", [
    ("시험1_미래예측", "exp1", 1485), ("시험2_미관측집단", "exp2", 315), ("시험3_새질문", "exp3", 132)])
def test_cell_errors_match_result_tables(cells, experiment, adapter, n):
    rows = result_rows(experiment, BLLOSSOM)
    assert len(rows) == n
    for r in rows:
        cell = cells[adapter][(r["qkey"], r["attribute"], r["group"])]
        assert cell["error"] == pytest.approx(float(r["wd_ft"]), abs=1e-9)
        assert cell["zero_shot_error"] == pytest.approx(float(r["wd_zeroshot"]), abs=1e-9)
        assert cell["lower_bound"] == pytest.approx(float(r["lb_cell"]), abs=1e-9)
        if adapter != "exp1":
            assert cell["reference"]["error"] == pytest.approx(float(r["wd_ft_exp1"]), abs=1e-9)


def test_held_out_flags_match_result_tables(service, cells):
    # 시험 2 결과표에 있는 집단이 곧 학습에서 뺀 집단이다
    held = {(r["attribute"], r["group"]) for r in result_rows("시험2_미관측집단", BLLOSSOM)}
    assert len(held) == 7
    for (qkey, attr, group), cell in cells["exp2"].items():
        assert cell["held_out"] == ((attr, group) in held)
    assert not any(c["held_out"] for c in cells["exp1"].values())
    # 시험 3 결과표에 있는 문항이 곧 학습에서 뺀 문항이다
    held_q = {r["qkey"] for r in result_rows("시험3_새질문", BLLOSSOM)}
    assert len(held_q) == 4
    for q in service.store.target_questions:
        assert service.prediction(q.qkey, "SEX", "exp3")["question"]["held_out"] == (q.qkey in held_q)


def test_distributions_are_normalised(service, cells):
    for adapter in config.ADAPTERS:
        for (qkey, _, _), cell in cells[adapter].items():
            n = len(service.store.questions[qkey].options)
            for key in ("predicted", "actual", "zero_shot"):
                assert len(cell[key]) == n
                assert sum(cell[key]) == pytest.approx(1.0, abs=1e-6)
                assert min(cell[key]) >= 0


def test_summary_is_mean_of_groups(service):
    p = service.prediction("q189_W2025", "TENURE", "exp1")
    errors = [g["error"] for g in p["groups"]]
    assert p["summary"]["error"] == pytest.approx(sum(errors) / len(errors))
    assert p["summary"]["error_readable"]["unit"] == "points"
    assert p["summary"]["error_readable"]["value"] == pytest.approx(p["summary"]["error"] * 10)


def test_zero_to_ten_scale_is_shown_with_its_own_labels(service):
    # 자료에는 1~11로 들어 있지만 보기는 0~10이다. 평균은 보기에 적힌 숫자로 보여야 한다.
    p = service.prediction("q189_W2025", "TENURE", "exp1")
    assert p["question"]["options"][0].startswith("0") and p["question"]["options"][-1].startswith("10")
    assert (p["question"]["scale_min"], p["question"]["scale_max"]) == (0, 10)
    assert p["question"]["option_values"] == [float(v) for v in range(11)]
    g = p["groups"][0]
    assert g["actual_mean"] == pytest.approx(sum(v * s for v, s in zip(range(11), g["actual"])))


def test_out_of_scale_option_is_excluded_from_mean(service):
    p = service.prediction("q73_1_W2025", "SEX", "exp1")
    assert p["question"]["options"][-1] == "잘 모르겠다"
    assert p["question"]["option_values"][-1] is None
    g = p["groups"][0]
    inside = g["actual"][:-1]
    assert g["actual_mean"] == pytest.approx(sum((i + 1) * s for i, s in enumerate(inside)) / sum(inside))


def test_nominal_question_has_no_means(service):
    p = service.prediction("q50_W2025", "AGE", "exp1")
    assert p["summary"]["error_metric"] == "tv"
    assert p["summary"]["order_corr"] is None
    assert all(g["predicted_mean"] is None and g["actual_mean"] is None for g in p["groups"])
    assert p["summary"]["error_readable"]["unit"] == "percent"


def test_last_year_comparison_uses_2024_actual(service):
    p = service.prediction("q132_1_W2025", "TENURE", "exp1")
    for g in p["groups"]:
        years = [h["year"] for h in g["history"]]
        assert 2024 in years
        last = next(h for h in g["history"] if h["year"] == 2024)
        assert g["last_year_error"] == last["error"]
    assert p["summary"]["error"] < p["summary"]["last_year_error"]


def test_two_group_attribute_has_no_order_correlation(service):
    assert service.prediction("q189_W2025", "SEX", "exp1")["summary"]["order_corr"] is None


def test_unknown_inputs_are_rejected(service):
    with pytest.raises(NotFound):
        service.prediction("q189_W2024", "TENURE", "exp1")   # 2025년 문항만 예측 대상이다
    with pytest.raises(NotFound):
        service.prediction("q189_W2025", "AGE_X_SEX", "exp1")
    with pytest.raises(NotFound):
        service.prediction("q189_W2025", "TENURE", "exp9")


def test_question_search(service):
    assert len(service.questions()) == 113
    hits = service.questions("버스 인프라")
    assert hits[0]["qkey"] == "q51_1_W2025"
    assert service.questions("없는말없는말") == []


def test_scenarios_point_at_real_cells(service):
    ids = set()
    for s in SCENARIOS:
        ids.add(s["id"])
        if s["panel"]:
            p = service.prediction(s["panel"]["qkey"], s["panel"]["attribute"], s["panel"]["adapter"])
            assert p["groups"]
            if s["panel"]["compare"] == "reference":
                assert all(g["reference"] for g in p["groups"])
            if s["panel"]["compare"] == "last_year":
                assert p["summary"]["last_year_error"] is not None
    assert len(ids) == len(SCENARIOS)


def test_brief_numbers_equal_panel_numbers(service):
    """대화 LLM에 주는 요약과 화면에 가는 자료가 같은 숫자여야 한다 (PRD A3)."""
    p = service.prediction("q120_1_W2025", "TENURE", "exp3")
    b = service.prediction_brief("q120_1_W2025", "TENURE", "exp3")
    assert b["평균_오차"] == round(p["summary"]["error"], 4)
    assert b["기본_모델_오차"] == round(p["summary"]["reference_error"], 4)
    assert b["문항"]["이_어댑터가_학습에서_뺀_문항"] is True
    for row, g in zip(b["집단별"], p["groups"]):
        assert row["집단"] == g["group"]
        assert row["분포_오차"] == round(g["error"], 4)
        assert row["예측_평균"] == round(g["predicted_mean"], 2)
        assert row["평균_차이_예측_빼기_실측"] == round(g["predicted_mean"] - g["actual_mean"], 2)
        assert row["분포_오차_읽기"] == f"7점 척도에서 약 {g['error'] * 6:.2f}점"
