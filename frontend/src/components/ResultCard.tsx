import { useState } from "react";
import { ArrowUpRight, Database, Radio, ShieldCheck } from "lucide-react";
import type { CompareKey, PanelState } from "../types";
import type { Theme } from "../lib/color";
import { cardStats, previewSeries, previewStats, seriesFor } from "../lib/format";
import { isLiveMode, sourceLabel, sourceTitle } from "../lib/labels";
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

/**
 * 답변 안에 들어가는 결과 카드: 문항, 한 줄 요약, 예측과 지난 조사의 막대.
 * 이 시연은 2025년 조사를 아직 하지 않았다고 놓는다. 2025년 결과(가려 두었던 값)와 오차는 '검증 보기'에서만 보인다.
 */
export function ResultCard({ state, compare, theme, compact, active, onDetail }: Props) {
  const { data: p, error } = usePrediction(state);
  const [verify, setVerify] = useState(false);

  if (error) return <div className="rc rc-error">{error}</div>;
  if (!p)
    return (
      <div className={`rc rc-loading${compact ? " is-compact" : ""}`} aria-busy="true">
        {isLiveMode() ? "모델을 돌리는 중…" : "불러오는 중…"}
      </div>
    );

  const { main, versus } = verify ? cardStats(p, compare) : previewStats(p);
  const series = verify ? seriesFor(p, new Set(compare ? [compare] : [])) : previewSeries(p, compare);
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
            {p.attribute.label}별 · {p.adapter.label}
            {main ? ` · ${main.value}` : ""}
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
            className={`rc-tag rc-source${live ? " is-live" : ""}`}
            title={sourceTitle(live)}
          >
            {live ? <Radio size={12} aria-hidden /> : <Database size={12} aria-hidden />}
            {sourceLabel(live)}
          </span>
          {q.held_out && <span className="badge badge-held">학습에서 뺀 문항</span>}
        </div>
        {question}
      </header>

      {verify && (
        <p className="rc-verify-note">
          <ShieldCheck size={13} aria-hidden /> 검증: 이 AI를 만들 때 가려 두었던 {p.predict_year}년 조사 결과와 견줍니다.
        </p>
      )}

      {(main || versus) && (
      <div className="rc-stats">
        {main && (
          <div className="rc-stat is-main">
            <span className="rc-stat-label">{main.label}</span>
            <strong>{main.value}</strong>
          </div>
        )}
        {versus && (
          <div className="rc-stat">
            <span className="rc-stat-label">{versus.label}</span>
            <strong>{versus.value}</strong>
          </div>
        )}
      </div>
      )}

      <MiniChart p={p} series={series} theme={theme} />

      <footer className="rc-foot">
        <button
          className={`rc-open rc-verify${verify ? " is-on" : ""}`}
          onClick={() => setVerify((v) => !v)}
          aria-pressed={verify}
          title={`이 AI를 만들 때 가려 두었던 ${p.predict_year}년 조사 결과와 견줘 본다`}
        >
          <ShieldCheck size={14} aria-hidden /> {verify ? "검증 닫기" : "검증 보기"}
        </button>
        <button className="rc-open" onClick={() => onDetail(state, compare)}>
          자세히 보기 <ArrowUpRight size={14} aria-hidden />
        </button>
      </footer>
    </article>
  );
}
