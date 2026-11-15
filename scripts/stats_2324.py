"""가상 보고서를 쓸 때 쓰는 숫자 조회 도구. 2023·2024년 실측만 보여 준다 (2025년 자료는 읽지 않는다).

    python scripts/stats_2324.py list [검색어]          # 2023·2024년 문항 목록 (변수 | 연도 | 종류 | 문구)
    python scripts/stats_2324.py show <변수> [속성]     # 그 문항의 연도별·집단별 분포(%)와 평균

속성: SEX AGE HOUSING TENURE EMPLOYMENT COMMUTE RESIDENCE_YEARS RELIGION (생략하면 전부)
"""
import csv
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "수원서베이_학습결과_정리본" / "07_데이터"
YEARS = ("2023", "2024")
ATTR = {"SEX": "성별", "AGE": "연령", "HOUSING": "주택유형", "TENURE": "점유형태", "EMPLOYMENT": "고용형태",
        "COMMUTE": "통근·통학", "RESIDENCE_YEARS": "거주기간", "RELIGION": "종교"}


def codebook():
    with open(SRC / "suwon/codebook/문항표_v2.csv", encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["연도"] in YEARS]


def text_of(r):
    return r["문항줄기"] + (f" - {r['항목명']}" if r["행렬형"] == "1" and r["항목명"] else "")


def cmd_list(query):
    for r in sorted(codebook(), key=lambda r: (r["변수"], r["연도"])):
        t = text_of(r)
        if all(tok in t for tok in query):
            print(f"{r['변수']} | {r['연도']} | {r['척도']} | {t}")


def cmd_show(var, attrs):
    meta = {r["연도"]: r for r in codebook() if r["변수"] == var}
    if not meta:
        sys.exit(f"2023·2024년에 없는 변수: {var}")
    counts = {}
    with open(SRC / "suwon/processed/집단별_응답자수_v2.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            counts[(r["연도"], r["attribute"], r["group"])] = r["n"]
    rows = {}
    with open(SRC / "suwon-v2-full/processed/suwon-v2-full.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            year = r["qkey"].rsplit("_W", 1)[1]
            if year in YEARS and r["qkey"] == f"{var}_W{year}":
                rows.setdefault(year, []).append(r)
    for year in YEARS:
        if year not in meta:
            print(f"\n[{year}년] 이 문항은 조사하지 않았다")
            continue
        m = meta[year]
        options = json.loads(m["보기"])[:-1]            # 마지막 '무응답'은 뺀다
        print(f"\n[{year}년] {text_of(m)}  ({m['척도']}, 응답 {m['응답수']}명)")
        print("  보기:", " / ".join(f"{i + 1}={o}" for i, o in enumerate(options)))
        for r in rows.get(year, []):
            if attrs and r["attribute"] not in attrs:
                continue
            dist = json.loads(r["responses"])
            ordinal = json.loads(r["ordinal"])
            pct = " ".join(f"{v * 100:.1f}" for v in dist)
            mean = ""
            if m["척도"] == "순서형":
                keep = [(d, o) for d, o in zip(dist, ordinal) if o >= 0]
                s = sum(d for d, _ in keep)
                mean = f" | 평균(자료의 순서 값 기준) {sum(d * o for d, o in keep) / s:.2f}" if s else ""
            n = counts.get((year, r["attribute"], r["group"]), "?")
            print(f"  {ATTR[r['attribute']]} · {r['group']} (n={n}): {pct}{mean}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "list":
        cmd_list(sys.argv[2:])
    elif len(sys.argv) >= 3 and sys.argv[1] == "show":
        cmd_show(sys.argv[2], set(sys.argv[3:]))
    else:
        sys.exit(__doc__)
