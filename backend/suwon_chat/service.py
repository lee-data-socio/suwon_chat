"""화면과 대화 LLM에 줄 자료를 만든다. 화면에 나오는 모든 숫자는 여기서 계산한다 (PRD P2)."""
from __future__ import annotations

import re
from statistics import mean

from . import config
from .metrics import display_dist, distance, is_nominal, mean_score, pearson, scale_range
from .predict import PredictionBackend
from .store import Question, Store


class NotFound(Exception):
    pass


def label_offset(q: Question) -> float:
    """보기 표기의 숫자와 자료의 순서 값이 일정하게 어긋나는 문항의 보정값.

    '0(최하)'~'10(최상)' 문항은 자료에 1~11로 들어 있다. 화면에는 보기에 적힌 숫자로 평균을 보여야 한다.
    """
    offsets = set()
    for label, o in zip(q.options, q.ordinal):
        if o < 0:
            continue
        m = re.match(r"\s*(-?\d+)", label)
        if not m:
            return 0.0
        offsets.add(int(m.group(1)) - o)
    return offsets.pop() if len(offsets) == 1 else 0.0


def _avg(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


class Service:
    def __init__(self, store: Store, backend: PredictionBackend):
        self.store = store
        self.backend = backend
        self._question_error = self._question_errors()

    # ---- 목록 -------------------------------------------------------------
    def meta(self) -> dict:
        year = config.PREDICT_YEAR
        return {
            "predict_year": year,
            "source": self.backend.source,
            "adapters": [
                {"id": k, "label": v["label"], "short": v["short"], "description": v["description"],
                 "held_out_groups": [{"attribute": a, "attribute_label": config.ATTRIBUTE_LABELS[a], "group": g}
                                     for a, g in v["held_out_groups"]],
                 "held_out_questions": v["held_out_questions"]}
                for k, v in config.ADAPTERS.items()],
            "attributes": [
                {"id": a, "label": config.ATTRIBUTE_LABELS[a],
                 "groups": [{"group": g, "n": self.store.group_n.get((year, a, g))} for g in self.store.groups[a]]}
                for a in config.ATTRIBUTE_LABELS],
            "self_check": self.store.self_check,
        }

    def _question_errors(self) -> dict[str, float]:
        """문항마다 기본 모델의 평균 오차 (33개 집단 평균). 문항 고르는 화면에서 쓴다."""
        out = {}
        for q in self.store.target_questions:
            errs = []
            for attr, groups in self.store.groups.items():
                for g in groups:
                    row = self.store.predictions[("exp1", q.qkey, attr, g)]
                    errs.append(distance(row["gold"], row["lora"], q.ordinal))
            out[q.qkey] = mean(errs)
        return out

    def question_summary(self, q: Question) -> dict:
        return {
            "qkey": q.qkey, "text": q.text, "stem": q.stem, "item": q.item, "survey_no": q.survey_no,
            "scale_type": q.scale_type, "options": q.options, "three_year": q.three_year,
            "error": round(self._question_error[q.qkey], 4),
            "error_metric": "tv" if is_nominal(q.ordinal) else "wd",
            "held_out_in": [k for k, v in config.ADAPTERS.items() if q.qkey in v["held_out_questions"]],
        }

    def questions(self, query: str = "", limit: int | None = None) -> list[dict]:
        items = self.store.target_questions
        tokens = [t for t in query.lower().split() if t]
        if tokens:
            # 검색어가 많이 들어간 문항부터. 같으면 조사표 순서를 지킨다.
            scored = []
            for i, q in enumerate(items):
                hay = f"{q.text} {' '.join(q.options)} {q.qkey}".lower()
                hits = sum(t in hay for t in tokens)
                if hits:
                    scored.append((-hits, i, q))
            scored.sort(key=lambda x: x[:2])
            items = [q for _, _, q in scored]
        out = [self.question_summary(q) for q in items]
        return out[:limit] if limit else out

    # ---- 예측과 실측 ------------------------------------------------------
    def prediction(self, qkey: str, attribute: str, adapter: str) -> dict:
        q = self.store.questions.get(qkey)
        if q is None or q.year != config.PREDICT_YEAR:
            raise NotFound(f"{config.PREDICT_YEAR}년 문항이 아니다: {qkey}")
        if attribute not in self.store.groups:
            raise NotFound(f"없는 속성: {attribute}")
        if adapter not in config.ADAPTERS:
            raise NotFound(f"없는 어댑터: {adapter}")
        spec = config.ADAPTERS[adapter]
        n = len(q.ordinal)
        nominal = is_nominal(q.ordinal)
        rng = scale_range(q.ordinal)
        off = label_offset(q)

        def score(dist):
            m = mean_score(dist, q.ordinal)
            return None if m is None else m + off

        held_groups = {g for a, g in spec["held_out_groups"] if a == attribute}
        prev_q = {y: self.store.previous(q, y) for y in (2023, 2024)}
        preds = self.backend.predict(qkey, attribute, adapter)
        reference = self.backend.predict(qkey, attribute, "exp1") if adapter != "exp1" else None

        groups = []
        for g in self.store.groups[attribute]:
            gold = self.store.predictions[("exp1", qkey, attribute, g)]["gold"]
            lora, base = preds[g]["lora"], preds[g]["base"]
            history = []
            for y, pq in prev_q.items():
                dist = self.store.actual.get((pq.qkey, attribute, g)) if pq else None
                if dist is not None:
                    history.append({"year": y, "dist": display_dist(dist, n), "mean": score(dist),
                                    "error": distance(gold, dist, q.ordinal)})
            last = next((h for h in history if h["year"] == config.PREDICT_YEAR - 1), None)
            row = {
                "group": g,
                "n": self.store.group_n.get((config.PREDICT_YEAR, attribute, g)),
                "held_out": g in held_groups,
                "predicted": display_dist(lora, n),
                "actual": display_dist(gold, n),
                "zero_shot": display_dist(base, n),
                "predicted_mean": score(lora),
                "actual_mean": score(gold),
                "zero_shot_mean": score(base),
                "error": distance(gold, lora, q.ordinal),
                "zero_shot_error": distance(gold, base, q.ordinal),
                "last_year_error": last["error"] if last else None,
                "lower_bound": self.store.lower_bound.get((qkey, attribute, g)),
                "history": history,
                "reference": None,
            }
            if reference is not None:
                ref = reference[g]["lora"]
                row["reference"] = {"adapter": "exp1", "predicted": display_dist(ref, n),
                                    "predicted_mean": score(ref),
                                    "error": distance(gold, ref, q.ordinal)}
            groups.append(row)

        def last_mean(r):
            h = next((h for h in r["history"] if h["year"] == config.PREDICT_YEAR - 1), None)
            return h["mean"] if h else None

        pm = [r["predicted_mean"] for r in groups]
        am = [r["actual_mean"] for r in groups]
        lm = [last_mean(r) for r in groups]
        has_means = not nominal and all(v is not None for v in pm + am)
        width = (rng[1] - rng[0]) if rng else None
        error_mean = mean(r["error"] for r in groups)
        summary = {
            # 순서형·이분형은 WD, 명목형은 TV. 둘 다 0~1이고 작을수록 가깝다.
            "error_metric": "tv" if nominal else "wd",
            "error": error_mean,
            # 오차를 사람이 읽는 단위로 바꾼 값. 순서형은 척도 점수, 이분형·명목형은 퍼센트포인트.
            "error_readable": self._readable(q, error_mean, width),
            "zero_shot_error": mean(r["zero_shot_error"] for r in groups),
            "last_year_error": _avg([r["last_year_error"] for r in groups]) if all(
                r["last_year_error"] is not None for r in groups) else None,
            "lower_bound": _avg([r["lower_bound"] for r in groups]),
            "reference_error": mean(r["reference"]["error"] for r in groups) if reference is not None else None,
            # 집단 평균의 높낮이가 실제와 같은 방향인지 (집단 3개 이상일 때만)
            "order_corr": pearson(pm, am) if has_means else None,
            "last_year_order_corr": pearson(lm, am) if has_means and all(v is not None for v in lm) else None,
            # 집단 사이 평균의 벌어진 정도 (최대 − 최소)
            "spread_predicted": (max(pm) - min(pm)) if has_means else None,
            "spread_actual": (max(am) - min(am)) if has_means else None,
        }
        return {
            "source": self.backend.source,
            "predict_year": config.PREDICT_YEAR,
            "adapter": {"id": adapter, "label": spec["label"], "short": spec["short"],
                        "description": spec["description"]},
            "attribute": {"id": attribute, "label": config.ATTRIBUTE_LABELS[attribute]},
            "question": {
                **self.question_summary(q),
                "ordinal": q.ordinal,
                # 보기마다 평균 계산에 쓰는 점수 ('잘 모르겠다'처럼 척도 밖인 보기는 None)
                "option_values": [None if o < 0 or nominal else o + off for o in q.ordinal],
                "scale_min": rng[0] + off if rng else None,
                "scale_max": rng[1] + off if rng else None,
                "scored": q.scale_type == "순서형" and q.three_year,
                "held_out": qkey in spec["held_out_questions"],
                "history_years": [y for y, pq in prev_q.items() if pq is not None],
            },
            "groups": groups,
            "summary": summary,
        }

    @staticmethod
    def _readable(q: Question, error: float, width: float | None) -> dict:
        if q.scale_type == "순서형" and width:
            points = int(round(width)) + 1
            return {"unit": "points", "value": error * width, "scale_points": points,
                    "text": f"{points}점 척도에서 약 {error * width:.2f}점"}
        return {"unit": "percent", "value": error * 100, "scale_points": None,
                "text": f"약 {error * 100:.1f}%p"}

    # ---- 대화 LLM에 줄 요약 ------------------------------------------------
    def prediction_brief(self, qkey: str, attribute: str, adapter: str) -> dict:
        """대화 LLM에는 차트 자료 전체가 아니라 문장에 쓸 요약만 준다. 차트 자료는 화면으로 직접 간다."""
        p = self.prediction(qkey, attribute, adapter)
        s, q = p["summary"], p["question"]
        r4 = lambda v: None if v is None else round(v, 4)
        r2 = lambda v: None if v is None else round(v, 2)
        return {
            "문항": {"qkey": q["qkey"], "문구": q["text"], "종류": q["scale_type"], "보기": q["options"],
                    "채점범위": q["scored"], "이_어댑터가_학습에서_뺀_문항": q["held_out"]},
            "속성": p["attribute"]["label"],
            "어댑터": p["adapter"]["label"],
            "자료_출처": "사전 계산" if p["source"] == "precomputed" else "실시간",
            "오차_지표": "WD(0~1, 작을수록 가까움)" if s["error_metric"] == "wd" else "TV(0~1, 작을수록 가까움)",
            "평균_오차": r4(s["error"]),
            "평균_오차_읽기": s["error_readable"]["text"],
            "학습_전_모델_오차": r4(s["zero_shot_error"]),
            "작년_실측을_그대로_쓴_경우_오차": r4(s["last_year_error"]),
            "사람_응답의_표본오차_하한": r4(s["lower_bound"]),
            "기본_모델_오차": r4(s["reference_error"]),
            "집단_평균_높낮이_상관": r2(s["order_corr"]),
            "집단_평균_벌어짐_예측": r2(s["spread_predicted"]),
            "집단_평균_벌어짐_실측": r2(s["spread_actual"]),
            "집단별": [
                {"집단": g["group"], "응답자수": g["n"], "학습에서_뺀_집단": g["held_out"],
                 "예측_평균": r2(g["predicted_mean"]), "실측_평균": r2(g["actual_mean"]),
                 "오차": r4(g["error"]),
                 "기본_모델_오차": r4(g["reference"]["error"]) if g["reference"] else None,
                 "예측_분포": [round(v, 3) for v in g["predicted"]],
                 "실측_분포": [round(v, 3) for v in g["actual"]]}
                for g in p["groups"]],
        }
