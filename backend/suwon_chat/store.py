"""로컬 DB를 읽어 메모리에 올려 둔다. 1만 5천 행 규모라 통째로 올려도 가볍다."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class Question:
    qkey: str
    var: str
    year: int
    scale_type: str          # 순서형 | 이분형 | 명목형
    survey_no: str
    stem: str
    item: str
    text: str
    options: list[str]       # 무응답을 뗀 보기
    ordinal: list[float]
    three_year: bool
    same_options: bool


class Store:
    def __init__(self, db_path: Path = config.DB_PATH):
        if not db_path.exists():
            raise FileNotFoundError(
                f"색인 DB가 없다: {db_path}\n먼저 `python -m suwon_chat.build_index`를 돌린다.")
        con = sqlite3.connect(db_path)
        self.questions: dict[str, Question] = {}
        for r in con.execute("SELECT qkey,var,year,scale_type,survey_no,stem,item,text,options,ordinal,"
                             "three_year,same_options FROM questions"):
            ordinal = json.loads(r[9])
            self.questions[r[0]] = Question(
                qkey=r[0], var=r[1], year=r[2], scale_type=r[3], survey_no=r[4], stem=r[5], item=r[6],
                text=r[7], options=json.loads(r[8])[: len(ordinal)], ordinal=ordinal,
                three_year=bool(r[10]), same_options=bool(r[11]))
        # 집단: 속성 → [집단] (자료에 적힌 순서), (연도, 속성, 집단) → 응답자 수
        self.groups: dict[str, list[str]] = {}
        self.group_n: dict[tuple, int] = {}
        for year, attr, grp, _pos, n in con.execute(
                "SELECT year,attribute,grp,pos,n FROM groups ORDER BY attribute,pos"):
            self.group_n[(year, attr, grp)] = n
            if grp not in self.groups.setdefault(attr, []):
                self.groups[attr].append(grp)
        self.actual = {(q, a, g): json.loads(d)
                       for q, a, g, d in con.execute("SELECT qkey,attribute,grp,dist FROM actual")}
        self.predictions = {(ad, q, a, g): {"gold": json.loads(gold), "base": json.loads(base), "lora": json.loads(lora)}
                            for ad, q, a, g, gold, base, lora in con.execute(
                                "SELECT adapter,qkey,attribute,grp,gold,base,lora FROM predictions")}
        self.lower_bound = {(q, a, g): lb for q, a, g, lb in con.execute("SELECT qkey,attribute,grp,lb FROM lower_bound")}
        row = con.execute("SELECT value FROM meta WHERE key='self_check'").fetchone()
        self.self_check = json.loads(row[0]) if row else None
        con.close()
        self.target_questions = sorted(
            (q for q in self.questions.values() if q.year == config.PREDICT_YEAR), key=lambda q: _natural(q.qkey))

    def previous(self, q: Question, year: int) -> Question | None:
        """같은 문항의 이전 연도. 문구·보기·순서 값이 그 해와 똑같을 때만 견줄 수 있다.

        문항표의 `연도간보기동일`은 3개 연도를 한꺼번에 본 표시라, 2023년만 다른 문항의
        2024년 비교까지 막는다. 그래서 두 해를 직접 견준다.
        """
        p = self.questions.get(f"{q.var}_W{year}")
        if p is None or p.text != q.text or p.options != q.options or p.ordinal != q.ordinal:
            return None
        return p


def _natural(qkey: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", qkey)]
