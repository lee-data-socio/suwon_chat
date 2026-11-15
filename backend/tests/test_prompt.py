"""프롬프트가 정리본의 평가용 프롬프트와 글자 하나까지 같은지 본다."""
import csv

import pytest

from suwon_chat import config
from suwon_chat.prompt import build_prompt, with_refusal


@pytest.mark.parametrize("dataset", ["suwon-v2-full", "suwon-v2-exp2", "suwon-v2-exp3"])
def test_prompts_equal_the_evaluation_prompts(service, dataset):
    store = service.store
    csv.field_size_limit(10**9)
    path = config.SOURCE_DIR / "07_데이터" / dataset / "processed" / "opnqa_QA_test.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3729
    for r in rows:
        q = store.questions[r["qkey"]]
        mine = build_prompt(store.steering[r["attribute"]], store.groups[r["attribute"]], r["group"], q.text, q.all_options)
        assert mine == r["input_prompt"], (r["qkey"], r["attribute"], r["group"])


def test_prompt_shape():
    p = build_prompt("귀하의 성별은 무엇입니까?", ["남", "여"], "여", "만족하십니까? ", ["예", "아니오", "무응답"])
    assert p == ("Question: 귀하의 성별은 무엇입니까?\nA. 남\nB. 여\n\nAnswer: B. 여\n\n"
                 "Answer the following question keeping in mind your previous answers.\n"
                 "Question: 만족하십니까?\nA. 예\nB. 아니오\nC. 무응답\nAnswer as a choice between A.,B.,C.\n\nAnswer:")


def test_refusal_is_appended_once():
    assert with_refusal(["예", " 아니오 ", ""]) == ["예", "아니오", "무응답"]
    assert with_refusal(["예", "아니오", "무응답"]) == ["예", "아니오", "무응답"]


def test_too_many_or_too_few_options_are_rejected():
    with pytest.raises(ValueError):
        build_prompt("q", ["a", "b"], "a", "문항", ["하나"])
    with pytest.raises(ValueError):
        build_prompt("q", ["a", "b"], "a", "문항", [str(i) for i in range(27)])
