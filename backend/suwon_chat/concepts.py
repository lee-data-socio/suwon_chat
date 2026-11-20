"""개념 사전으로 문항을 찾는다 (질문 이해용 검색).

정책 담당자는 조사표 문구로 묻지 않는다("월세 사는 사람들 형편 어때?"). 그래서 문항 문구에
`concepts/개념사전.md`의 개념 정의와 일상 표현을 덧붙여 색인하고, 질문과 의미가 가까운 문항을 찾는다.
임베딩 모델은 보고서 검색과 같은 로컬 모델이다.

담당자식 질문 40개로 잰 결과 (scripts/eval_question_search.py, 2026-11-20): 1순위 적중이
단어 검색 8/20·8/20, 문항 문구만 의미 검색 12/20·18/20, 개념 사전을 덧붙인 의미 검색 16/20·19/20.

    python -m suwon_chat.concepts         # 개념 사전 + 문항 → data/concepts_index.json, data/concepts_vectors.npy
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import config

Embed = Callable[[list[str], str], np.ndarray]
TOP_K = 5


class ConceptError(ValueError):
    pass


@dataclass
class Concept:
    name: str
    specs: list[str]                 # 변수. 끝이 _이면 그것으로 시작하는 변수 모두
    definition: str
    expressions: str
    per_item: dict[str, str] = field(default_factory=dict)

    def covers(self, var: str) -> bool:
        return any(var == s or (s.endswith("_") and var.startswith(s)) for s in self.specs)


def parse_concepts(path: Path) -> list[Concept]:
    if not path.exists():
        raise ConceptError(f"개념 사전이 없다: {path}")
    concepts = []
    for sec in re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)[1:]:
        lines = [l.strip() for l in sec.strip().splitlines()]

        def get(key: str) -> str:
            return next((l.split(":", 1)[1].strip() for l in lines if l.startswith(key + ":")), "")

        specs = [s.strip() for s in get("문항").split(",") if s.strip()]
        if not specs or not get("정의"):
            raise ConceptError(f"'{lines[0]}': 문항과 정의가 있어야 한다")
        per = {m.group(1): m.group(2).strip() for m in (re.match(r"- (q\w+): (.+)", l) for l in lines) if m}
        concepts.append(Concept(lines[0], specs, get("정의"), get("일상 표현"), per))
    if not concepts:
        raise ConceptError(f"개념이 하나도 없다: {path}")
    return concepts


def concept_for(concepts: list[Concept], var: str) -> Concept | None:
    return next((c for c in concepts if c.covers(var)), None)


def item_text(question, concept: Concept | None) -> str:
    """색인에 넣는 글: 문항 문구와 보기, 그리고 개념 사전의 정의와 일상 표현."""
    text = f"{question.text} / 보기: {', '.join(question.options[:7])}"
    if concept is None:
        return text
    extra = concept.per_item.get(question.var, "")
    return f"{text} / {concept.name}. {concept.definition} 일상 표현: {concept.expressions} {extra}".strip()


def build_index(questions: list, concepts_path: Path, embed: Embed, index_path: Path, vectors_path: Path) -> dict:
    concepts = parse_concepts(concepts_path)
    entries, texts, missing = [], [], []
    for q in questions:
        c = concept_for(concepts, q.var)
        if c is None:
            missing.append(q.var)
        entries.append({"qkey": q.qkey, "var": q.var, "concept": c.name if c else None,
                        "definition": c.definition if c else None})
        texts.append(item_text(q, c))
    vectors = embed(texts, "passage")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    index_path.write_text(json.dumps({"model": getattr(embed, "model_name", "custom"), "entries": entries},
                                     ensure_ascii=False), encoding="utf-8")
    return {"questions": len(entries), "concepts": len(concepts), "missing": missing}


class QuestionIndex:
    """만들어 둔 색인으로 문항을 찾는다. 색인이 없으면 `available`이 거짓이다."""

    def __init__(self, index_path: Path, vectors_path: Path, embed: Embed | None = None):
        self.available = index_path.exists() and vectors_path.exists()
        self.embed = embed
        self.entries: list[dict] = []
        self.vectors = np.zeros((0, 1), np.float32)
        if self.available:
            self.entries = json.loads(index_path.read_text(encoding="utf-8"))["entries"]
            self.vectors = np.load(vectors_path)

    def concept(self, qkey: str) -> dict | None:
        """문항의 개념 이름과 정의 (답변 첫머리에 쓴다)"""
        e = next((e for e in self.entries if e["qkey"] == qkey), None)
        return {"개념": e["concept"], "정의": e["definition"]} if e and e["concept"] else None

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        """[{qkey, concept, definition, score}] 가까운 순. 임베딩을 못 쓰면 빈 목록."""
        if not self.available or self.embed is None or not query.strip():
            return []
        qvec = self.embed([query.strip()], "query")
        if qvec.shape[1] != self.vectors.shape[1]:
            return []                      # 임베딩 모델을 쓰지 못했다
        scores = (qvec @ self.vectors.T)[0]
        return [{**self.entries[i], "score": round(float(scores[i]), 4)} for i in np.argsort(-scores)[:k]]


if __name__ == "__main__":
    from .reports import LocalEmbedder
    from .store import Store
    store = Store()
    try:
        result = build_index(store.target_questions, config.CONCEPTS_FILE, LocalEmbedder(local_only=False),
                             config.CONCEPTS_INDEX, config.CONCEPTS_VECTORS)
    except ConceptError as e:
        print(f"개념 사전 색인 실패: {e}", file=sys.stderr)
        sys.exit(1)
    if result["missing"]:
        print("개념 사전에 없는 문항 (문항 문구만으로 찾는다):", ", ".join(result["missing"]), file=sys.stderr)
    print("문항 검색 색인 완료:", json.dumps(result, ensure_ascii=False))
