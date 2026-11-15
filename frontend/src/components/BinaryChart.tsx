import type { Prediction } from "../types";
import type { SeriesDef } from "../lib/format";
import { distOf, pct } from "../lib/format";
import { useSize } from "../lib/hooks";
import { isMain, lineSizes } from "../lib/layout";
import { GroupLabel } from "./GroupLabel";
import { ErrorCell } from "./OrdinalChart";
import { SeriesMark } from "./SeriesMark";
import { Hint, useTip } from "./Tooltip";

const TICKS = [0, 25, 50, 75, 100];

/** 이분형 문항: 첫 보기를 고른 비율을 집단마다 예측·실측 막대로 나란히 놓는다. 가로축은 0~100%로 고정한다. */
export function BinaryChart({ p, series }: { p: Prediction; series: SeriesDef[] }) {
  const q = p.question;
  const [rowsRef, rowsSize] = useSize<HTMLDivElement>();
  const tip = useTip();
  const rowH = p.groups.length ? rowsSize.height / p.groups.length : 0;
  const sizes = lineSizes(rowH, series, 18);

  return (
    <div className={`chart binary${series.length >= 4 ? " wide-err" : ""}`}>
      <div className="chart-head chart-grid binary-grid">
        <div className="col-title">집단</div>
        <div className="col-head">
          <div className="col-title">
            <span className="layer-no">①</span> ‘{q.options[0]}’라고 답한 비율
            <span className="col-note">
              나머지는 ‘{q.options[1]}’ · 가로축은 0~100%
            </span>
          </div>
          <div className="axis binary-axis">
            <div className="axis-ticks">
              {TICKS.map((t) => (
                <span key={t} style={{ left: `${t}%` }}>
                  {t}%
                </span>
              ))}
            </div>
          </div>
        </div>
        <div className="col-title col-right">
          <Hint text="2025년 실측과의 거리. 이분형 문항에서는 두 비율의 차이와 같다.">오차</Hint>
        </div>
      </div>
      <div className="chart-rows" ref={rowsRef}>
        {p.groups.map((g) => (
          <div className={`chart-row chart-grid binary-grid${g.held_out ? " is-held" : ""}`} key={g.group}>
            <GroupLabel g={g} />
            <div className="bars">
              {series.map((s, si) => {
                const d = distOf(p, g, s.key);
                const v = d ? d[0] : null;
                const content = (
                  <div>
                    <div className="tip-title">
                      {g.group} · {q.options[0]}
                    </div>
                    {series.map((t) => {
                      const dd = distOf(p, g, t.key);
                      return (
                        dd && (
                          <div className="tip-row" key={t.key}>
                            <SeriesMark series={t.key} size={9} />
                            <span className="tip-name">{t.name}</span>
                            <strong>{pct(dd[0], 1)}</strong>
                          </div>
                        )
                      );
                    })}
                  </div>
                );
                return (
                  <div
                    className={`bar-line${isMain(s.key) ? "" : " is-minor"}`}
                    key={s.key}
                    style={{ height: sizes[si].line }}
                  >
                    <span className="bar-tag" style={{ width: 56, fontSize: sizes[si].font }}>
                      {s.tag}
                    </span>
                    <div
                      className="share-plot"
                      style={{ height: sizes[si].bar }}
                      tabIndex={0}
                      onPointerMove={(e) => tip.show(e, content)}
                      onPointerLeave={tip.hide}
                      onFocus={(e) => tip.show(e.currentTarget, content)}
                      onBlur={tip.hide}
                    >
                      {TICKS.map((t) => (
                        <i className="share-grid" key={t} style={{ left: `${t}%` }} />
                      ))}
                      {v != null && (
                        <>
                          <div className={`share-bar series-${s.key}`} style={{ width: `${v * 100}%` }} />
                          <span className="share-value" style={{ left: `${v * 100}%`, fontSize: Math.min(12, sizes[si].font + 0.5) }}>
                            {pct(v, 1)}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
            <ErrorCell g={g} series={series} />
          </div>
        ))}
      </div>
    </div>
  );
}
