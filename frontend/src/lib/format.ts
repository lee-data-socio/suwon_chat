import type { CompareKey, GroupRow, Prediction } from "../types";
import { adapterLabel, adapterShort, REFERENCE_ID } from "./labels";

export const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;
/** 숫자 표기. 반올림하면 0이 되는 아주 작은 양수는 "0.001 미만"처럼 쓴다 (0.000이라고 쓰면 차이가 없다는 뜻으로 읽힌다). */
export const num = (v: number | null | undefined, digits = 2) => {
  if (v == null) return "–";
  const unit = 10 ** -digits;
  if (v > 0 && v < unit / 2) return `${unit.toFixed(digits)} 미만`;
  if (Math.abs(v) < unit / 2) return (0).toFixed(digits); // "−0.00"이 나오지 않게 한다
  return v.toFixed(digits);
};
export const signed = (v: number | null | undefined, digits = 2) => {
  if (v == null) return "–";
  if (Math.abs(v) < 10 ** -digits / 2) return (0).toFixed(digits);
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
};
/** 상관. 반올림해서 ±1.00이 되는 값은 완전히 일치한 것처럼 보이므로 "0.99 이상"으로 쓴다. */
export const corr = (v: number | null | undefined) => {
  if (v == null) return "–";
  if (v >= 0.995) return "+0.99 이상";
  if (v <= -0.995) return "−0.99 이하";
  return signed(v);
};
export const count = (n: number | null) => (n == null ? "" : `${n.toLocaleString("ko-KR")}명`);

/** '1(전혀 신뢰하지 않는다)' → '전혀 신뢰하지 않는다'. 숫자뿐이면 그대로 둔다. */
export function optionWords(label: string): string {
  const m = label.match(/^\s*-?\d+\s*\((.+)\)\s*$/);
  return m ? m[1] : label;
}

/** 화면에 나오는 계열. 예측과 실측은 늘 있고 나머지는 켜고 끈다. */
export type SeriesKey = "predicted" | "actual" | CompareKey;

export interface SeriesDef {
  key: SeriesKey;
  /** 막대 옆에 붙는 짧은 이름 */
  tag: string;
  /** 범례·표에 쓰는 이름 */
  name: string;
  /** 가는 막대로 그린다. 비우면 예측·실측만 굵게 */
  minor?: boolean;
}

export function seriesFor(p: Prediction, compare: Set<CompareKey>): SeriesDef[] {
  const year = p.predict_year;
  const isBase = p.adapter.id === REFERENCE_ID;
  const out: SeriesDef[] = [
    { key: "predicted", tag: "예측", name: `${p.adapter.label}의 예측` },
    { key: "actual", tag: "검증", name: `${year}년 결과(검증용)` },
  ];
  if (compare.has("reference") && !isBase) out.push({ key: "reference", tag: adapterShort(REFERENCE_ID), name: `${adapterLabel(REFERENCE_ID)}의 예측` });
  if (compare.has("last_year") && hasLastYear(p))
    out.push({ key: "last_year", tag: "작년", name: `${year - 1}년 조사` });
  if (compare.has("zero_shot")) out.push({ key: "zero_shot", tag: "학습 전", name: "학습 전 모델의 예측" });
  return out;
}

/**
 * 답변 속 카드의 기본 계열: 예측과 지난 조사. 이 시연은 2025년 조사를 아직 하지 않았다고 놓으므로
 * 2025년 결과는 '검증 보기'를 눌렀을 때만 보인다. 빼고 학습한 AI면 보고 학습한 AI의 예측도 함께 둔다.
 */
export function previewSeries(p: Prediction, compare: CompareKey | null): SeriesDef[] {
  const year = p.predict_year;
  const out: SeriesDef[] = [{ key: "predicted", tag: "예측", name: `${year}년 예측` }];
  if (hasLastYear(p)) out.push({ key: "last_year", tag: `${year - 1}`, name: `${year - 1}년 조사`, minor: false });
  if (compare === "reference" && p.adapter.id !== REFERENCE_ID)
    out.push({ key: "reference", tag: adapterShort(REFERENCE_ID), name: `${adapterLabel(REFERENCE_ID)}의 예측` });
  return out;
}

export const hasLastYear = (p: Prediction) => p.question.history_years.includes(p.predict_year - 1);

const lastYear = (p: Prediction, g: GroupRow) => g.history.find((h) => h.year === p.predict_year - 1);

export function distOf(p: Prediction, g: GroupRow, key: SeriesKey): number[] | null {
  switch (key) {
    case "predicted":
      return g.predicted;
    case "actual":
      return g.actual;
    case "zero_shot":
      return g.zero_shot;
    case "reference":
      return g.reference?.predicted ?? null;
    case "last_year":
      return lastYear(p, g)?.dist ?? null;
  }
}

export function meanOf(p: Prediction, g: GroupRow, key: SeriesKey): number | null {
  switch (key) {
    case "predicted":
      return g.predicted_mean;
    case "actual":
      return g.actual_mean;
    case "zero_shot":
      return g.zero_shot_mean;
    case "reference":
      return g.reference?.predicted_mean ?? null;
    case "last_year":
      return lastYear(p, g)?.mean ?? null;
  }
}

/** 2025년 실측과의 거리. 실측 자신은 없다. */
export function errorOf(g: GroupRow, key: SeriesKey): number | null {
  switch (key) {
    case "predicted":
      return g.error;
    case "actual":
      return null;
    case "zero_shot":
      return g.zero_shot_error;
    case "reference":
      return g.reference?.error ?? null;
    case "last_year":
      return g.last_year_error;
  }
}

export const metricName = (m: "wd" | "tv") => (m === "wd" ? "분포 거리(WD)" : "분포 거리(TV)");

/**
 * 오차를 사람이 읽는 단위로 바꾼다 (순서형은 척도 점수, 이분형은 %p, 명목형은 %).
 * 전체 평균 오차는 서버가 준 문구(`summary.error_readable.text`)를 그대로 쓰고,
 * 이 함수는 서버 문구가 없는 값(비교 대상, 집단 하나)에만 쓴다. 같은 환산식을 따른다.
 */
export function readable(p: Prediction, error: number | null | undefined, withScale = false): string | null {
  if (error == null) return null;
  const r = p.summary.error_readable;
  if (r.unit === "points" && p.question.scale_max != null && p.question.scale_min != null) {
    const v = error * (p.question.scale_max - p.question.scale_min);
    return `${withScale && r.scale_points ? `${r.scale_points}점 척도에서 ` : ""}약 ${v.toFixed(2)}점`;
  }
  return p.question.scale_type === "이분형" ? `약 ${(error * 100).toFixed(1)}%p` : `약 ${(error * 100).toFixed(1)}%`;
}

export interface Stat {
  label: string;
  value: string;
}

/** 결과 카드의 한 줄 요약: 예측 오차 하나와, 켜 둔 비교 대상 하나. */
export function cardStats(p: Prediction, compare: CompareKey | null): { main: Stat; versus: Stat | null } {
  const held = p.groups.filter((g) => g.held_out);
  const one = held.length === 1 ? held[0] : null;
  const main: Stat = one
    ? { label: `학습에서 뺀 ‘${one.group}’의 예측 오차`, value: readable(p, one.error, true) ?? "–" }
    : { label: p.question.held_out ? "학습에서 뺀 문항의 예측 오차" : "예측 오차", value: p.summary.error_readable.text };
  let versus: Stat | null = null;
  const s = p.summary;
  if (compare === "last_year" && s.last_year_error != null)
    versus = { label: `작년(${p.predict_year - 1}) 실측을 그대로 쓰면`, value: readable(p, s.last_year_error)! };
  else if (compare === "zero_shot") versus = { label: "학습 전 모델은", value: readable(p, s.zero_shot_error)! };
  else if (compare === "reference" && s.reference_error != null) {
    if (one?.reference) versus = { label: "이 집단을 보고 학습하면", value: readable(p, one.reference.error)! };
    else if (p.question.held_out) versus = { label: "이 문항을 보고 학습하면", value: readable(p, s.reference_error)! };
    else versus = { label: `${adapterLabel(REFERENCE_ID)}는`, value: readable(p, s.reference_error)! };
  }
  return { main, versus };
}

/** 집단들의 값 범위를 한 줄로: 순서형은 평균, 이분형은 첫 보기 비율. 명목형은 한 줄로 줄이지 않는다 */
function spread(p: Prediction, key: SeriesKey): string | null {
  const kind = p.question.scale_type;
  const vals = p.groups
    .map((g) => (kind === "순서형" ? meanOf(p, g, key) : kind === "이분형" ? (distOf(p, g, key)?.[0] ?? null) : null))
    .filter((v): v is number => v != null);
  if (!vals.length || vals.length !== p.groups.length) return null;
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  if (kind === "순서형") return lo === hi ? `평균 ${num(lo)}점` : `평균 ${num(lo)}~${num(hi)}점`;
  return `‘${p.question.options[0]}’ ${pct(lo, 1)}~${pct(hi, 1)}`;
}

/** 검증 전 카드의 한 줄 요약: 예측과 지난 조사의 집단 범위. 2025년 결과와 오차는 쓰지 않는다. */
export function previewStats(p: Prediction): { main: Stat | null; versus: Stat | null } {
  const year = p.predict_year;
  const pred = spread(p, "predicted");
  const last = hasLastYear(p) ? spread(p, "last_year") : null;
  return {
    main: pred ? { label: `${year}년 예측 · 집단별`, value: pred } : null,
    versus: last ? { label: `${year - 1}년 조사 · 집단별`, value: last } : null,
  };
}

/** 가상 문서의 제목 앞에 붙은 '[시연용 가상 문서]'는 표시(배지)로 따로 보이므로 뗀다. */
export const cleanTitle = (title: string) => title.replace(/^\s*\[[^\]]*가상 문서[^\]]*\]\s*/, "");

/** 참고 문서 대목을 카드에 넣을 한 줄 글로 만든다 (소제목 표시와 줄바꿈을 걷어낸다). */
export const plainSnippet = (text: string) =>
  text
    .split("\n")
    .filter((l) => !/^\s*#{1,6}\s/.test(l))
    .join(" ")
    .replace(/\*\*/g, "")
    .replace(/\s+/g, " ")
    .trim();
