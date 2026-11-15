"""참고 문서(보고서) 색인과 검색.

문서는 `reports/*.md`에 있다. 머리말(id, title, summary, topics)과 `<!-- page N -->`로 나눈 쪽으로 되어 있고,
쪽 하나가 검색 단위다. 임베딩 모델은 로컬에서 돈다. 문서 내용이 밖으로 나가지 않는다.

    python -m suwon_chat.reports          # reports/ → data/reports_index.json, data/reports_vectors.npy

참고 문서는 예측에 쓰이지 않는다. 화면에서도 예측·실측과 따로 보인다 (PRD P1).
"""
from __future__ import annotations

import json
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import config

EMBED_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 3
MAX_PER_DOC = 2          # 한 문서에서 너무 많이 뽑히지 않게 한다
# 관련 없는 문서를 억지로 띄우지 않기 위한 문턱. 이 임베딩 모델은 무관한 글끼리도 0.8 안팎이 나온다.
# 값은 색인한 문서로 문항 113개의 검색 결과를 훑어보고 정했다 (README의 "참고 문서" 절).
MIN_SCORE = 0.87         # 이보다 낮으면 관련 문서가 없다고 본다 (0.857~0.869 구간은 무관한 문서가 절반쯤 섞였다)
MARGIN = 0.025           # 1등과 이만큼 넘게 벌어진 대목은 버린다
PAGE_MARK = re.compile(r"^<!--\s*page\s+(\d+)\s*-->\s*$", re.M)
SYNTHETIC_TAG = "[시연용 가상 문서]"

Embed = Callable[[list[str], str], np.ndarray]   # (글 목록, "query" | "passage") → 길이 1로 맞춘 벡터


class ReportError(Exception):
    pass


@dataclass
class Document:
    doc_id: str
    title: str
    summary: str
    topics: list[str]
    pages: list[dict]        # [{"page": 1, "text": "..."}]

    @property
    def synthetic(self) -> bool:
        return self.title.startswith(SYNTHETIC_TAG)

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "summary": self.summary, "topics": self.topics,
                "synthetic": self.synthetic, "pages": self.pages}


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        raise ReportError(f"{path.name}: 머리말(--- … ---)이 없다")
    head, body = {}, m.group(2)
    for line in m.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            head[key.strip()] = value.strip()
    for key in ("id", "title"):
        if not head.get(key):
            raise ReportError(f"{path.name}: 머리말에 {key}가 없다")
    unquote = lambda s: s[1:-1] if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'" else s
    try:
        topics = json.loads(head.get("topics", "[]"))
    except json.JSONDecodeError:
        topics = []
    marks = list(PAGE_MARK.finditer(body))
    if not marks:
        raise ReportError(f"{path.name}: 쪽 표시(<!-- page N -->)가 없다")
    pages = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        text = body[mark.end():end].strip()
        if text:
            pages.append({"page": int(mark.group(1)), "text": text})
    if not pages:
        raise ReportError(f"{path.name}: 내용이 없다")
    return Document(doc_id=unquote(head["id"]), title=unquote(head["title"]),
                    summary=unquote(head.get("summary", "")), topics=topics, pages=pages)


def load_documents(folder: Path) -> list[Document]:
    docs = [parse_document(p) for p in sorted(folder.glob("*.md"))]
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ReportError("문서 id가 겹친다")
    return docs


def plain(text: str) -> str:
    """검색과 카드에 쓰는 글: 마크다운 표시를 뗀다."""
    return re.sub(r"\s+", " ", re.sub(r"^#+\s*", "", text, flags=re.M)).strip()


# ---- 임베딩 -----------------------------------------------------------------
class LocalEmbedder:
    """다국어 임베딩 모델을 로컬에서 돌린다. 처음 쓸 때 모델을 올린다."""

    def __init__(self, model: str = EMBED_MODEL, local_only: bool = True):
        self.model_name, self.local_only = model, local_only
        self._lock = threading.Lock()
        self._tok = self._model = None

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                from transformers import AutoModel, AutoTokenizer
                kwargs = {"local_files_only": self.local_only}
                self._tok = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
                self._model = AutoModel.from_pretrained(self.model_name, **kwargs).eval()

    def __call__(self, texts: list[str], kind: str) -> np.ndarray:
        import torch
        self.load()
        out = []
        for i in range(0, len(texts), 16):
            batch = self._tok([f"{kind}: {t}" for t in texts[i:i + 16]], padding=True, truncation=True,
                              max_length=512, return_tensors="pt")
            with torch.no_grad():
                hidden = self._model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            vec = (hidden * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(vec, dim=-1).numpy())
        return np.concatenate(out).astype(np.float32)


# ---- 색인 -------------------------------------------------------------------
def _rank(scores: np.ndarray, chunks: list[dict], k: int = TOP_K) -> list[tuple[int, float]]:
    picked, per_doc = [], {}
    for i in np.argsort(-scores):
        doc = chunks[i]["doc_id"]
        if per_doc.get(doc, 0) >= MAX_PER_DOC:
            continue
        per_doc[doc] = per_doc.get(doc, 0) + 1
        picked.append((int(i), float(scores[i])))
        if len(picked) == k:
            break
    return picked


def build_index(folder: Path, questions: dict[str, str], embed: Embed,
                index_path: Path, vectors_path: Path) -> dict:
    """문서를 읽어 쪽마다 벡터를 만들고, 문항마다 가까운 대목을 미리 골라 둔다."""
    docs = load_documents(folder)
    if not docs:
        raise ReportError(f"문서가 없다: {folder}")
    chunks = [{"doc_id": d.doc_id, "page": p["page"]} for d in docs for p in d.pages]
    texts = [f"{d.title.replace(SYNTHETIC_TAG, '').strip()}. {plain(p['text'])}" for d in docs for p in d.pages]
    vectors = embed(texts, "passage")
    by_question = {}
    if questions:
        qkeys = list(questions)
        qvec = embed([questions[k] for k in qkeys], "query")
        scores = qvec @ vectors.T
        for row, qkey in zip(scores, qkeys):
            by_question[qkey] = [[i, round(s, 4)] for i, s in _rank(row, chunks)]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    index_path.write_text(json.dumps({"model": getattr(embed, "model_name", "custom"),
                                      "documents": [d.to_dict() for d in docs], "chunks": chunks,
                                      "by_question": by_question}, ensure_ascii=False), encoding="utf-8")
    return {"documents": len(docs), "pages": len(chunks), "questions": len(by_question)}


class ReportIndex:
    """만들어 둔 색인을 읽어 검색한다. 색인이 없으면 `available`이 거짓이고 검색 결과는 비어 있다."""

    def __init__(self, index_path: Path, vectors_path: Path, embed: Embed | None = None,
                 min_score: float = MIN_SCORE, margin: float = MARGIN):
        self.min_score, self.margin = min_score, margin
        self.available = index_path.exists() and vectors_path.exists()
        self.embed = embed
        self.documents: dict[str, dict] = {}
        self.chunks: list[dict] = []
        self.by_question: dict[str, list] = {}
        self.vectors = np.zeros((0, 1), np.float32)
        if self.available:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            self.documents = {d["doc_id"]: d for d in data["documents"]}
            self.chunks = data["chunks"]
            self.by_question = data["by_question"]
            self.vectors = np.load(vectors_path)

    def _item(self, i: int, score: float) -> dict:
        chunk = self.chunks[i]
        doc = self.documents[chunk["doc_id"]]
        page = next(p for p in doc["pages"] if p["page"] == chunk["page"])
        return {"doc_id": doc["doc_id"], "title": doc["title"], "summary": doc["summary"],
                "page": chunk["page"], "pages": len(doc["pages"]),
                "text": page["text"], "score": round(score, 4), "synthetic": doc["synthetic"]}

    def _keep(self, ranked: list) -> list[dict]:
        if not ranked or ranked[0][1] < self.min_score:
            return []
        top = ranked[0][1]
        return [self._item(i, s) for i, s in ranked if s >= self.min_score and s >= top - self.margin]

    def for_question(self, qkey: str) -> list[dict]:
        return self._keep(self.by_question.get(qkey, []))

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        if not self.available or self.embed is None or not query.strip():
            return []
        qvec = self.embed([query.strip()], "query")
        if qvec.shape[1] != self.vectors.shape[1]:
            return []                      # 임베딩 모델을 쓰지 못했다
        scores = (qvec @ self.vectors.T)[0]
        return self._keep(_rank(scores, self.chunks, k))

    def document(self, doc_id: str) -> dict | None:
        return self.documents.get(doc_id)

    def listing(self) -> list[dict]:
        return [{k: d[k] for k in ("doc_id", "title", "summary", "topics", "synthetic")} | {"pages": len(d["pages"])}
                for d in self.documents.values()]


if __name__ == "__main__":
    from .store import Store
    store = Store()
    try:
        embedder = LocalEmbedder(local_only=False)      # 색인을 만들 때만 모델을 내려받는다
        result = build_index(config.REPORTS_DIR, {q.qkey: q.text for q in store.target_questions}, embedder,
                             config.REPORTS_INDEX, config.REPORTS_VECTORS)
    except ReportError as e:
        print(f"보고서 색인 실패: {e}", file=sys.stderr)
        sys.exit(1)
    print("보고서 색인 완료:", json.dumps(result, ensure_ascii=False))
