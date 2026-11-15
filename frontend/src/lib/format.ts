import type { CompareKey, GroupRow, Prediction } from "../types";
import { adapterLabel, adapterShort, REFERENCE_ID } from "./labels";

export const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;
export const num = (v: number | null | undefined, digits = 2) => (v == null ? "–" : v.toFixed(digits));
export const signed = (v: number | null | undefined, digits = 2) =>
  v == null ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
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
}

export function seriesFor(p: Prediction, compare: Set<CompareKey>): SeriesDef[] {
  const year = p.predict_year;
  const isBase = p.adapter.id === REFERENCE_ID;
  const out: SeriesDef[] = [
    { key: "predicted", tag: "예측", name: `${p.adapter.label}의 예측` },
    { key: "actual", tag: "실측", name: `${year}년 실측` },
  ];
  if (compare.has("reference") && !isBase) out.push({ key: "reference", tag: adapterShort(REFERENCE_ID), name: `${adapterLabel(REFERENCE_ID)}의 예측` });
  if (compare.has("last_year") && hasLastYear(p))
    out.push({ key: "last_year", tag: "작년", name: `${year - 1}년 실측` });
  if (compare.has("zero_shot")) out.push({ key: "zero_shot", tag: "학습 전", name: "학습 전 모델의 예측" });
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
