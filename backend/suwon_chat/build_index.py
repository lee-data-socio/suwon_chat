"""정리본 폴더를 읽어 로컬 DB 하나(data/suwon.db)를 만든다. 정리본의 파일은 고치지 않는다.

    python -m suwon_chat.build_index

끝에 자체 점검을 돈다 (PRD A1): 행 수, 그리고 시험 1 전체 WD가 정리본의 값과 같은지.
"""
from __future__ import annotations

import ast
import csv
import json
import sqlite3
import sys
from pathlib import Path

from . import config
from .metrics import ordinal_emd

SCHEMA = """
CREATE TABLE questions (
    qkey TEXT PRIMARY KEY, var TEXT, year INTEGER, scale_type TEXT, survey_no TEXT,
    stem TEXT, item TEXT, text TEXT, options TEXT, ordinal TEXT,
    three_year INTEGER, same_options INTEGER
);
CREATE TABLE groups (year INTEGER, attribute TEXT, grp TEXT, pos INTEGER, n INTEGER,
    PRIMARY KEY (year, attribute, grp));
CREATE TABLE actual (qkey TEXT, attribute TEXT, grp TEXT, dist TEXT, refusal_rate REAL,
    PRIMARY KEY (qkey, attribute, grp));
CREATE TABLE predictions (adapter TEXT, qkey TEXT, attribute TEXT, grp TEXT,
    gold TEXT, base TEXT, lora TEXT, PRIMARY KEY (adapter, qkey, attribute, grp));
CREATE TABLE lower_bound (qkey TEXT, attribute TEXT, grp TEXT, lb REAL,
    PRIMARY KEY (qkey, attribute, grp));
CREATE TABLE steering (attribute TEXT PRIMARY KEY, question TEXT);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""

EXPECTED_ACTUAL_ROWS = 14850
EXPECTED_PRED_ROWS = 3729
EXPECTED_SCORED_CELLS = 1485
EXPECTED_EXP1_WD = 0.0413


class IndexFailed(Exception):
    """색인이나 자체 점검이 실패했다."""


def _require(ok: bool, message: str) -> None:
    # assert는 `python -O`에서 사라지므로 쓰지 않는다
    if not ok:
        raise IndexFailed(message)


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _find(src: Path, name: str) -> Path:
    hits = sorted(src.rglob(name))
    if not hits:
        raise FileNotFoundError(f"정리본에서 {name} 을 찾지 못했다: {src}")
    return hits[0]


def build(src: Path = config.SOURCE_DIR, db_path: Path = config.DB_PATH) -> dict:
    if not src.exists():
        raise FileNotFoundError(f"정리본 폴더가 없다: {src}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        report = _fill(con, src)
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    con.close()
    tmp.replace(db_path)
    return report


def _fill(con: sqlite3.Connection, src: Path) -> dict:
    con.executescript(SCHEMA)

    # 집단을 알려 주는 앞 질문 (실시간 예측에서 프롬프트를 만들 때 쓴다)
    with open(_find(src, "steering_prompts_suwon_v2.json"), encoding="utf-8") as f:
        con.executemany("INSERT INTO steering VALUES (?,?)", [(s["attribute"], s["qa_prompt"]) for s in json.load(f)])

    # 실측 분포 (3개 연도). 순서 값은 문항표가 아니라 이 파일에서 읽는다 (문항표는 명목형의 순서 값이 비어 있다).
    actual = _read_csv(src / "07_데이터/suwon-v2-full/processed/suwon-v2-full.csv")
    ordinal_by_q = {r["qkey"]: json.loads(r["ordinal"]) for r in actual}

    # 문항표
    qt = _read_csv(_find(src, "문항표_v2.csv"))
    years_by_var: dict[str, set] = {}
    for r in qt:
        years_by_var.setdefault(r["변수"], set()).add(int(r["연도"]))
    for r in qt:
        item = r["항목명"].strip()
        text = r["문항줄기"] + (f" - {item}" if r["행렬형"] == "1" and item else "")
        con.execute(
            "INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["qkey"], r["변수"], int(r["연도"]), r["척도"], r["조사표문항번호"], r["문항줄기"], item, text,
             json.dumps(json.loads(r["보기"]), ensure_ascii=False), json.dumps(ordinal_by_q[r["qkey"]]),
             int(years_by_var[r["변수"]] == {2023, 2024, 2025}), int(r["연도간보기동일"] == "1")),
        )

    # 집단과 응답자 수
    order = {}
    for r in _read_csv(_find(src, "demographics_suwon_v2.csv")):
        for i, g in enumerate(ast.literal_eval(r["group"])):
            order[(r["attribute"], g)] = i
    for r in _read_csv(_find(src, "집단별_응답자수_v2.csv")):
        con.execute("INSERT INTO groups VALUES (?,?,?,?,?)",
                    (int(r["연도"]), r["attribute"], r["group"], order[(r["attribute"], r["group"])], int(r["n"])))

    con.executemany("INSERT INTO actual VALUES (?,?,?,?,?)",
                    [(r["qkey"], r["attribute"], r["group"], json.dumps(json.loads(r["responses"])),
                      float(r["refusal_rate"])) for r in actual])

    # 어댑터별 2025년 예측
    suffix = f"_W{config.PREDICT_YEAR}"
    pred_counts = {}
    for adapter, spec in config.ADAPTERS.items():
        with open(src / spec["pred_file"], encoding="utf-8") as f:
            rows = json.load(f)["rows"]
        held_groups = {tuple(g) for g in spec["held_out_groups"]}
        held_questions = {q[: -len(suffix)] for q in spec["held_out_questions"]}
        for r in rows:
            if r["split"] == "test":
                continue
            # 학습에서 뺐다고 적은 집단·문항이 실제로 학습·검증 자료에 없는지 확인한다
            _require((r["attribute"], r["group"]) not in held_groups,
                     f"{adapter}: 뺐다고 적은 집단이 학습 자료에 있다 ({r['attribute']} {r['group']})")
            _require(r["qkey"].rsplit("_W", 1)[0] not in held_questions,
                     f"{adapter}: 뺐다고 적은 문항이 학습 자료에 있다 ({r['qkey']})")
        test = [r for r in rows if r["split"] == "test"]
        _require(all(r["qkey"].endswith(suffix) for r in test), f"{adapter}: 2025년이 아닌 시험 행이 있다")
        con.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?)",
                        [(adapter, r["qkey"], r["attribute"], r["group"],
                          json.dumps(r["gold"]), json.dumps(r["base"]), json.dumps(r["lora"])) for r in test])
        pred_counts[adapter] = len(test)

    # 사람 응답의 표본 오차 하한
    con.executemany("INSERT INTO lower_bound VALUES (?,?,?,?)",
                    [(r["qkey"], r["attribute"], r["group"], float(r["lb_mean"]))
                     for r in _read_csv(_find(src, "lower_bound_v2_2025_cells.csv"))])
    con.commit()

    report = self_check(con, len(actual), pred_counts)
    con.execute("INSERT INTO meta VALUES ('self_check', ?)", (json.dumps(report, ensure_ascii=False),))
    con.commit()
    return report


def self_check(con: sqlite3.Connection, n_actual: int, pred_counts: dict) -> dict:
    """PRD A1. 하나라도 어긋나면 멈춘다. 숫자가 틀린 채로 화면이 뜨면 안 된다."""
    _require(n_actual == EXPECTED_ACTUAL_ROWS, f"실측 행 수 {n_actual} (기대 {EXPECTED_ACTUAL_ROWS})")
    for adapter, n in pred_counts.items():
        _require(n == EXPECTED_PRED_ROWS, f"{adapter} 예측 행 수 {n} (기대 {EXPECTED_PRED_ROWS})")
    # 시험 1 채점 범위: 3개 연도에 모두 있는 순서형 문항 × 33개 집단
    rows = con.execute(
        """SELECT p.gold, p.lora, q.ordinal FROM predictions p JOIN questions q USING (qkey)
           WHERE p.adapter = 'exp1' AND q.scale_type = '순서형' AND q.three_year = 1""").fetchall()
    wds = [ordinal_emd(json.loads(g), json.loads(l), json.loads(o)) for g, l, o in rows]
    _require(len(wds) == EXPECTED_SCORED_CELLS, f"채점 칸 수 {len(wds)} (기대 {EXPECTED_SCORED_CELLS})")
    wd = sum(wds) / len(wds)
    _require(round(wd, 4) == EXPECTED_EXP1_WD, f"시험 1 WD {wd:.5f} (정리본 {EXPECTED_EXP1_WD})")
    return {"actual_rows": n_actual, "prediction_rows": pred_counts,
            "scored_cells": len(wds), "exp1_wd": round(wd, 4)}


if __name__ == "__main__":
    try:
        result = build()
    except (IndexFailed, FileNotFoundError) as e:
        print(f"색인 실패: {e}", file=sys.stderr)
        sys.exit(1)
    print("색인 완료:", json.dumps(result, ensure_ascii=False))
    print("DB:", config.DB_PATH)
