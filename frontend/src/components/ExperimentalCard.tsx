import { FlaskConical, Radio, TriangleAlert } from "lucide-react";
import type { Prediction } from "../types";
import type { Theme } from "../lib/color";
import type { SeriesDef } from "../lib/format";
import { sourceLabel, sourceTitle } from "../lib/labels";
import { MiniChart } from "./MiniChart";

/**
 * 목록에 없는 새 문항의 예측 카드. 실측이 없어 오차를 알 수 없다 (PRD P6).
 * 평소 결과 카드와 달리 점선 테두리와 '실험적' 표시를 쓰고, 예측 막대만 한 줄씩 그린다.
 * 상세 패널은 목록에 있는 문항만 다루므로 '자세히 보기'는 없다.
 */
export function ExperimentalCard({ p, theme }: { p: Prediction; theme: Theme }) {
  const series: SeriesDef[] = [{ key: "predicted", tag: "예측", name: `${p.adapter.label}의 예측` }];
  return (
    <article className="rc rc-exp">
      <header className="rc-head">
        <div className="rc-tags">
          <span className="rc-tag">{p.attribute.label}별</span>
          <span className="rc-tag" title={p.adapter.description}>
            {p.adapter.label}
          </span>
          <span className="rc-tag rc-source is-live" title={sourceTitle(true)}>
            <Radio size={12} aria-hidden />
            {sourceLabel(true)}
          </span>
          <span className="rc-tag rc-exp-tag">
            <FlaskConical size={12} aria-hidden />
            실험적
          </span>
        </div>
        <h3 className="rc-q" title={p.question.text}>
          {p.question.text}
        </h3>
      </header>

      <div className="rc-note" role="note">
        <TriangleAlert size={16} aria-hidden />
        <span>{p.note ?? "실측이 없는 새 문항입니다. 오차를 알 수 없는, 검증되지 않은 실험적 예측입니다."}</span>
      </div>

      <MiniChart p={p} series={series} theme={theme} />

      <footer className="rc-exp-foot">실측 없음 · 예측 막대만 보입니다</footer>
    </article>
  );
}
