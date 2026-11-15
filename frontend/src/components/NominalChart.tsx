import type { Prediction } from "../types";
import type { SeriesDef, SeriesKey } from "../lib/format";
import { count, distOf, num, pct } from "../lib/format";
import { useSize } from "../lib/hooks";
import { GroupBadges } from "./GroupLabel";
import { SeriesMark } from "./SeriesMark";
import { Hint, useTip } from "./Tooltip";

const MAIN: SeriesKey[] = ["predicted", "actual"];

/**
 * 명목형 문항: 보기(줄) × 집단(칸)으로 놓고 칸마다 예측·실측 막대를 나란히 그린다.
 * 모든 칸이 같은 눈금을 쓴다. 평균이 없는 문항이라 ② 평균 비교는 없다.
 */
export function NominalChart({ p, series }: { p: Prediction; series: SeriesDef[] }) {
  const q = p.question;
  const tip = useTip();
  const [bodyRef, bodySize] = useSize<HTMLDivElement>();
  const peak = Math.max(
    0.1,
    ...p.groups.flatMap((g) => series.flatMap((s) => distOf(p, g, s.key) ?? [])),
  );
  const top = Math.min(1, Math.ceil(peak * 10) / 10); // 칸 너비가 뜻하는 비율 (10% 단위로 올림)
  const rowH = q.options.length ? bodySize.height / q.options.length : 0;
  const weights = series.map((s) => (MAIN.includes(s.key) ? 1 : 0.7));
  const unit = Math.max(3, Math.min(9, (rowH - 12 - (series.length - 1) * 2) / weights.reduce((a, b) => a + b, 0)));
  const cols = { gridTemplateColumns: `minmax(120px, 1.25fr) repeat(${p.groups.length}, minmax(0, 1fr))` };

  return (
    <div className="chart nominal">
      <div className="chart-head nominal-grid" style={cols}>
        <div className="col-title">
          <span className="layer-no">①</span> 보기별 응답 비율
          <div className="col-note nominal-note">칸 너비 = {pct(top)}</div>
        </div>
        {p.groups.map((g) => (
          <div className={`nominal-group${g.held_out ? " is-held" : ""}`} key={g.group}>
            <div className="group-name" title={g.group}>
              {g.group}
            </div>
            <div className="group-sub">
              <span className="group-n">{count(g.n)}</span>
              <GroupBadges g={g} />
            </div>
          </div>
        ))}
      </div>
      <div className="nominal-body" ref={bodyRef}>
        {q.options.map((opt, oi) => (
          <div className="nominal-row nominal-grid" style={cols} key={oi}>
            <div className="nominal-option" title={opt}>
              {opt}
            </div>
            {p.groups.map((g) => {
              const content = (
                <div>
                  <div className="tip-title">
                    {g.group} · {opt}
                  </div>
                  {series.map((s) => {
                    const d = distOf(p, g, s.key);
                    return (
                      d && (
                        <div className="tip-row" key={s.key}>
                          <SeriesMark series={s.key} size={9} />
                          <span className="tip-name">{s.name}</span>
                          <strong>{pct(d[oi], 1)}</strong>
                        </div>
                      )
                    );
                  })}
                </div>
              );
              return (
                <div
                  className="nominal-cell"
                  key={g.group}
                  tabIndex={0}
                  onPointerMove={(e) => tip.show(e, content)}
                  onPointerLeave={tip.hide}
                  onFocus={(e) => tip.show(e.currentTarget, content)}
                  onBlur={tip.hide}
                >
                  {series.map((s, si) => {
                    const d = distOf(p, g, s.key);
                    return (
                      <div
                        key={s.key}
                        className={`share-bar series-${s.key}`}
                        style={{ width: `${d ? (d[oi] / top) * 100 : 0}%`, height: unit * weights[si] }}
                      />
                    );
                  })}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="nominal-foot nominal-grid" style={cols}>
        <div className="col-title">
          <Hint text="2025년 실측 분포와 어긋난 비율의 합(TV). 0이면 같고 1이면 전혀 겹치지 않는다.">오차</Hint>
        </div>
        {p.groups.map((g) => (
          <div className="err-main" key={g.group}>
            {num(g.error, 3)}
          </div>
        ))}
      </div>
    </div>
  );
}
