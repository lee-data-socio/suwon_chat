"""시청 담당자식 질문을 맞는 문항에 잇는지 잰다. 검색 방식 네 가지를 견준다.

    A. 단어 검색            지금의 search_questions (근거 단계 data)
    B. 의미 검색            문항 문구와 보기만 색인
    C. 의미 검색 + 보고서   B에 그 문항을 다룬 참고 문서의 제목·주제·요약을 덧붙임
    D. 의미 검색 + 개념 사전 B에 concepts/개념사전.md의 정의와 일상 표현을 덧붙임 (근거 단계 full)

    cd backend && python ../scripts/eval_question_search.py [--k 3] [--show]

질문과 정답은 scripts/question_search_cases.json에 있다. "기존" 묶음은 개념 사전을 쓰기 전에 만들었지만
같은 사람이 사전을 썼으므로 사전이 그 질문에 맞춰졌을 수 있다. "새" 묶음은 사전을 보지 않은 별도 에이전트가 만들었다.
임베딩 모델은 앱과 같은 로컬 모델(intfloat/multilingual-e5-small)이다.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PREDICT_BACKEND", "lookup")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402

from suwon_chat import config  # noqa: E402
from suwon_chat.concepts import concept_for, item_text, parse_concepts  # noqa: E402
from suwon_chat.predict import LookupBackend  # noqa: E402
from suwon_chat.reports import SYNTHETIC_TAG, LocalEmbedder, load_documents  # noqa: E402
from suwon_chat.service import Service  # noqa: E402
from suwon_chat.store import Store  # noqa: E402

CASES = Path(__file__).with_name("question_search_cases.json")


def matches(var: str, accept: list[str]) -> bool:
    return any(var == a or (a.endswith("_") and var.startswith(a)) for a in accept)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="상위 몇 개 안에 정답이 있으면 맞힌 것으로 볼지")
    ap.add_argument("--show", action="store_true", help="질문마다 결과를 보인다")
    args = ap.parse_args()

    store = Store()
    service = Service(store, LookupBackend(store))
    items = store.target_questions
    embed = LocalEmbedder()
    docs = load_documents(config.REPORTS_DIR)
    concepts = parse_concepts(config.CONCEPTS_FILE)

    def text_b(q):
        return item_text(q, None)

    def text_c(q):
        extra = [f"{d.title.replace(SYNTHETIC_TAG, '').strip()}. 주제: {', '.join(d.topics)}. {d.summary}"
                 for d in docs if d.covers(q.var)]
        return text_b(q) + (" / 관련 보고서: " + " ".join(extra) if extra else "")

    vectors = {"B": embed([text_b(q) for q in items], "passage"),
               "C": embed([text_c(q) for q in items], "passage"),
               "D": embed([item_text(q, concept_for(concepts, q.var)) for q in items], "passage")}

    data = json.loads(CASES.read_text(encoding="utf-8"))
    for name in ("기존", "새"):
        cases = data[name]
        qvec = embed([c[0] for c in cases], "query")
        top1 = dict.fromkeys("ABCD", 0)
        topk = dict.fromkeys("ABCD", 0)
        for i, (question, accept) in enumerate(cases):
            got = {"A": [store.questions[h["qkey"]].var for h in service.questions(question, limit=args.k)]}
            for m, v in vectors.items():
                got[m] = [items[j].var for j in np.argsort(-(v @ qvec[i]))[:args.k]]
            for m, vs in got.items():
                top1[m] += bool(vs) and matches(vs[0], accept)
                topk[m] += any(matches(v, accept) for v in vs)
            if args.show:
                marks = " ".join(f"{m}:{'O' if any(matches(v, accept) for v in vs) else 'X'}" for m, vs in got.items())
                print(f"  {marks} | {question} | 정답 {accept} | D {got['D']}")
        print(f"[{name} {len(cases)}개] 1순위 적중 " + ", ".join(f"{m} {top1[m]}" for m in "ABCD")
              + f" / 상위 {args.k}개 적중 " + ", ".join(f"{m} {topk[m]}" for m in "ABCD"))


if __name__ == "__main__":
    main()
