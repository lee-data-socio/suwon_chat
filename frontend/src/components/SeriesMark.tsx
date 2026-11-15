import type { SeriesKey } from "../lib/format";

/**
 * 계열마다 모양이 다른 표식. 색만으로 구분하지 않는다.
 * 예측 ● · 실측 ○ · 견주는 어댑터(전체 학습) ■ · 작년 실측 ◇ · 학습 전 모델 ▲
 */
export function SeriesMark({ series, size = 11, bar = false }: { series: SeriesKey; size?: number; bar?: boolean }) {
  // 막대 차트의 범례에서는 막대와 같은 모양으로 그린다
  if (bar) return <span className={`mark-bar series-${series}`} aria-hidden />;
  return <span className={`mark mark-${series}`} style={{ ["--mark" as string]: `${size}px` }} aria-hidden />;
}
