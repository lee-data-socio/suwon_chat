"""분포 사이의 거리와 평균 점수.

`ordinal_emd`는 정리본 `05_코드/03_재현/suwon_analysis/build_results_v2.py`의 같은 이름 함수를
그대로 옮긴 것이다. 보고서의 숫자와 화면의 숫자가 같으려면 이 함수가 같아야 한다.
"""
from __future__ import annotations

import numpy as np


def ordinal_emd(p, q, ordinal) -> float:
    """순서형 분포 두 개의 Wasserstein 거리 (0~1로 정규화)."""
    p = np.asarray(p, float); q = np.asarray(q, float); o = np.asarray(ordinal, float)
    n = min(len(p), len(q), len(o))
    p, q, o = p[:n], q[:n], o[:n]
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p, q = p / p.sum(), q / q.sum()
    if o.max() == o.min():
        return 0.0
    idx = np.argsort(o); o, p, q = o[idx], p[idx], q[idx]
    nn = np.where(o >= 0)[0]
    if len(nn) == 0:
        return 0.0
    if nn[0] > 0:
        o, p, q = o[nn[0]:], p[nn[0]:], q[nn[0]:]
        if p.sum() == 0 or q.sum() == 0:
            return 0.0
        p, q = p / p.sum(), q / q.sum()
    c1, c2 = np.cumsum(p), np.cumsum(q)
    return float(np.sum(np.abs(c1[:-1] - c2[:-1]) * (o[1:] - o[:-1])) / (o.max() - o.min()))


def is_nominal(ordinal) -> bool:
    """순서 값이 모두 같으면 순서가 없는 문항이다."""
    o = [v for v in ordinal if v >= 0]
    return len(o) < 2 or max(o) == min(o)


def display_dist(dist, n_options: int) -> list[float]:
    """화면에 보일 분포: 무응답 칸을 떼고 합이 1이 되게 한다."""
    d = np.asarray(dist[:n_options], float)
    s = d.sum()
    if s <= 0:
        return [0.0] * n_options
    return [float(x) for x in d / s]


def tv_distance(p, q) -> float:
    """순서가 없는 문항에 쓰는 거리: 두 분포가 어긋난 비율의 합 (0~1)."""
    n = min(len(p), len(q))
    a = np.asarray(p[:n], float); b = np.asarray(q[:n], float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def distance(p, q, ordinal) -> float:
    """문항 종류에 맞는 거리. 순서형·이분형은 WD, 명목형은 TV.

    예측 파일의 분포는 끝에 무응답 칸이 붙어 있다. WD는 순서 값 길이만큼만 쓰므로,
    TV도 같은 길이로 잘라 화면의 막대(무응답을 뗀 분포)와 같은 값이 나오게 한다.
    """
    if is_nominal(ordinal):
        n = len(ordinal)
        return tv_distance(p[:n], q[:n])
    return ordinal_emd(p, q, ordinal)


def mean_score(dist, ordinal) -> float | None:
    """척도 평균. '모름'처럼 순서 값이 음수인 보기는 뺀다. 명목형은 평균이 없다."""
    if is_nominal(ordinal):
        return None
    n = min(len(dist), len(ordinal))
    d = np.asarray(dist[:n], float); o = np.asarray(ordinal[:n], float)
    keep = o >= 0
    if d[keep].sum() <= 0:
        return None
    return float((d[keep] * o[keep]).sum() / d[keep].sum())


def scale_range(ordinal) -> tuple[float, float] | None:
    o = [v for v in ordinal if v >= 0]
    if len(o) < 2 or max(o) == min(o):
        return None
    return float(min(o)), float(max(o))


def pearson(x, y) -> float | None:
    """집단 평균의 높낮이가 맞는지 보는 상관. 집단이 3개 미만이면 뜻이 없어 내지 않는다."""
    if len(x) < 3 or len(x) != len(y):
        return None
    a = np.asarray(x, float); b = np.asarray(y, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])
