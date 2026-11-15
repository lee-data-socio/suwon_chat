import { ArrowUpRight, Database, Radio } from "lucide-react";
import type { CompareKey, PanelState } from "../types";
import type { Theme } from "../lib/color";
import { cardStats, seriesFor } from "../lib/format";
import { usePrediction } from "../lib/usePrediction";
import { MiniChart } from "./MiniChart";

interface Props {
  state: PanelState;
  compare: CompareKey | null;
  theme: Theme;
  /** 상세 패널이 열려 채팅 열이 좁을 때는 차트 없이 한 줄로 줄인다 */
  compact: boolean;
  /** 상세 패널이 지금 이 장면을 보이고 있다 */
  active: boolean;
  onDetail: (state: PanelState, compare: CompareKey | null) => void;
}

/** 답변 안에 들어가는 결과 카드: 문항, 한 줄 요약, 예측과 실측의 막대. */
export function ResultCard({ state, compare, theme, compact, active, onDetail }: Props) {
  const { data: p, error } = usePrediction(state);

  if (error) return <div className="rc rc-error">{error}</div>;
  if (!p) return <div className="rc rc-loading" aria-busy="true" />;

  const { main, versus } = cardStats(p, compare);
  const series = seriesFor(p, new Set(compare ? [compare] : []));
  const live = p.source === "live";
  const q = p.question;

  const question = (
    <h3 className="rc-q" title={q.text}>
      {q.item ? (
        <>
          <span className="rc-stem">{q.stem}</span>
          {q.item.replace(/^-\s*/, "")}
        </>
      ) : (
        q.text
      )}
    </h3>
  );

  if (compact)
    return (
      <article className={`rc is-compact${active ? " is-active" : ""}`}>
        <div className="rc-compact-main">
          {question}
          <div className="rc-compact-sub">
            {p.attribute.label}별 · {p.adapter.label} · {main.value}
          </div>
        </div>
        <button className="rc-open" onClick={() => onDetail(state, compare)} disabled={active}>
          {active ? "보는 중" : "자세히 보기"}
        </button>
      </article>
    );

  return (
    <article className="rc">
      <header className="rc-head">
        <div className="rc-tags">
          <span className="rc-tag">{p.attribute.label}별</span>
          <span className="rc-tag" title={p.adapter.description}>
            {p.adapter.label}
          </span>
          <span
            className="rc-tag rc-source"
            title={live ? "모델을 지금 돌려 얻은 예측이다." : "미리 계산해 저장해 둔 예측이다. 모델을 지금 돌린 값이 아니다."}
          >
            {live ? <Radio size={12} aria-hidden /> : <Database size={12} aria-hidden />}
            {live ? "실시간" : "사전 계산"}
          </span>
          {q.held_out && <span className="badge badge-held">학습에서 뺀 문항</span>}
        </div>
        {question}
      </header>

      <div className="rc-stats">
        <div className="rc-stat is-main">
          <span className="rc-stat-label">{main.label}</span>
          <strong>{main.value}</strong>
        </div>
        {versus && (
          <div className="rc-stat">
            <span className="rc-stat-label">{versus.label}</span>
            <strong>{versus.value}</strong>
          </div>
        )}
      </div>

      <MiniChart p={p} series={series} theme={theme} />

      <footer className="rc-foot">
        <button className="rc-open" onClick={() => onDetail(state, compare)}>
          자세히 보기 <ArrowUpRight size={14} aria-hidden />
        </button>
      </footer>
    </article>
  );
}
