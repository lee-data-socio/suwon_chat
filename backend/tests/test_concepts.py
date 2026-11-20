"""개념 사전과 문항 검색 색인"""
from suwon_chat import config
from suwon_chat.concepts import concept_for, parse_concepts


def test_concept_names_are_unique():
    names = [c.name for c in parse_concepts(config.CONCEPTS_FILE)]
    assert len(names) == len(set(names))


def test_dictionary_covers_all_113_questions(service):
    concepts = parse_concepts(config.CONCEPTS_FILE)
    missing = [q.var for q in service.store.target_questions if concept_for(concepts, q.var) is None]
    assert missing == []
    # 하위 항목 표현은 실제 있는 변수에만 붙는다
    vars_ = {q.var for q in service.store.target_questions}
    assert all(v in vars_ for c in concepts for v in c.per_item)


def test_index_returns_concept_and_definition(concept_index):
    assert concept_index.concept("q189_W2025")["개념"].startswith("주관적 계층 인식")
    found = concept_index.search("가구 부채 빚 대출 가계부채")
    assert found[0]["qkey"] == "q28_W2025" and found[0]["definition"]


def test_index_without_embedder_returns_nothing(concept_index):
    from suwon_chat.concepts import QuestionIndex
    idx = QuestionIndex.__new__(QuestionIndex)
    idx.__dict__.update(concept_index.__dict__, embed=None)
    assert idx.search("형편") == []
