import type { SeriesDef, SeriesKey } from "./format";

const MAIN: SeriesKey[] = ["predicted", "actual"];
export const isMain = (key: SeriesKey) => MAIN.includes(key);

export interface LineSize {
  /** 한 줄이 차지하는 높이 */
  line: number;
  /** 그 안의 막대 굵기 */
  bar: number;
  /** 막대 옆 이름표 글자 크기 */
  font: number;
}

/**
 * 한 집단의 줄 높이 안에 계열 막대들을 나눠 넣는다.
 * 예측·실측은 굵게, 함께 보는 것은 가늘게. 줄이 좁아지면 막대와 글자가 함께 줄어든다.
 */
export function lineSizes(rowHeight: number, series: SeriesDef[], maxBar = 22): LineSize[] {
  const MIN_LINE = 9.5; // 이름표 글자가 겹치지 않는 가장 좁은 줄
  const weights = series.map((s) => (isMain(s.key) ? 1 : 0.72));
  const total = weights.reduce((a, b) => a + b, 0);
  const avail = Math.max(0, rowHeight - 10);
  let heights = weights.map((w) => (avail * w) / total);
  if (heights.some((h) => h < MIN_LINE)) {
    // 좁으면 가는 줄부터 최소 높이를 주고 남는 높이를 굵은 줄이 나눠 갖는다
    const minor = weights.filter((w) => w < 1).length;
    const main = weights.length - minor;
    const mainH = Math.max(MIN_LINE, (avail - minor * MIN_LINE) / Math.max(1, main));
    heights = weights.map((w) => (w < 1 ? MIN_LINE : mainH));
  }
  return heights.map((h, i) => {
    const cap = isMain(series[i].key) ? maxBar : maxBar * 0.6;
    const gap = h >= 14 ? 4 : 2;
    const bar = Math.max(4, Math.min(cap, h - gap));
    const line = Math.min(h, bar + gap + 2);
    return { line, bar, font: Math.max(9, Math.min(11.5, line - 0.5)) };
  });
}
