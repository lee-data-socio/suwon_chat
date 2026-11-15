import { useMemo } from "react";
import type { GroupRow, Prediction } from "../types";
import type { Theme } from "../lib/color";
import { nominalFills, ordinalSwatches, stackOrder } from "../lib/color";
import type { SeriesDef } from "../lib/format";
import { count, distOf, meanOf, num, pct } from "../lib/format";
import { isMain } from "../lib/layout";
import { GroupBadges } from "./GroupLabel";
import { ScaleLegend } from "./OrdinalChart";
import { SeriesMark } from "./SeriesMark";
import { useTip } from "./Tooltip";

interface Props {
  p: Prediction;
  series: SeriesDef[];
  theme: Theme;
}

/**
 * 답변 안에 들어가는 작은 차트. 집단마다 예측 막대와 실측 막대를 위아래로 붙여,
 * 두 막대가 얼마나 겹치는지만 읽히게 한다. 자세한 숫자는 막대에 마우스를 올리거나 '자세히 보기'에서 본다.
 */
export function MiniChart({ p, series, theme }: Props) {
  const q = p.question;
  const kind = q.scale_type;
  const tip = useTip();
  const swatches = useMemo(() => ordinalSwatches(q.option_values, theme), [q.option_values, theme]);
  const order = useMemo(
    () => (kind === "순서형" ? stackOrder(q.option_values) : q.options.map((_, i) => i)),
    [kind, q.option_values, q.options],
  );
  const fills = useMemo(
    () => (kind === "순서형" ? swatches.map((s) => s.fill) : nominalFills(q.options, theme)),
    [kind, swatches, q.options, theme],
  );

  const optionTip = (g: GroupRow, i: number) => (
    <div>
      <div className="tip-title">
        {g.group} · {q.options[i]}
      </div>
      {series.map((s) => {
        const d = distOf(p, g, s.key);
        return (
          d && (
            <div className="tip-row" key={s.key}>
              <SeriesMark series={s.key} size={9} />
              <span className="tip-name">{s.name}</span>
              <strong>{pct(d[i], 1)}</strong>
            </div>
          )
        );
      })}
    </div>
  );

  return (
    <div className={`mc mc-${kind === "순서형" ? "ordinal" : kind === "이분형" ? "binary" : "nominal"}`}>
      <div className="mc-top">
        {kind === "순서형" && <ScaleLegend p={p} swatches={swatches} order={order} />}
        {kind === "이분형" && (
          <div className="mc-note">
            ‘{q.options[0]}’라고 답한 비율 <span>나머지는 ‘{q.options[1]}’</span>
          </div>
        )}
        {kind === "명목형" && (
          <div className="mc-options">
            {q.options.map((o, i) => (
              <span className="legend-item" key={i}>
                <i style={{ background: fills[i] }} />
                {o}
              </span>
            ))}
          </div>
        )}
        {kind === "순서형" && <div className="mc-colhead">평균</div>}
      </div>

      <div className="mc-rows">
        {p.groups.map((g) => (
          <div className={`mc-group${g.held_out ? " is-held" : ""}`} key={g.group}>
            <div className="mc-label">
              <div className="mc-name" title={g.group}>
                {g.group}
              </div>
              <div className="mc-sub">
                <span>{count(g.n)}</span>
                <GroupBadges g={g} />
              </div>
            </div>
            <div className="mc-lines">
              {series.map((s) => {
                const d = distOf(p, g, s.key);
                const minor = !isMain(s.key);
                return (
                  <div className={`mc-line${minor ? " is-minor" : ""}`} key={s.key}>
                    <span className="mc-tag">{s.tag}</span>
                    {kind === "이분형" ? (
                      <div
                        className="mc-track"
                        tabIndex={0}
                        onPointerMove={(e) => tip.show(e, optionTip(g, 0))}
                        onPointerLeave={tip.hide}
                        onFocus={(e) => tip.show(e.currentTarget, optionTip(g, 0))}
                        onBlur={tip.hide}
                      >
                        <i className="mc-half" />
                        {d && <div className={`mc-fill series-${s.key}`} style={{ width: `${d[0] * 100}%` }} />}
                      </div>
                    ) : (
                      <div className="mc-stack" role="img" aria-label={`${g.group} ${s.name}`}>
                        {d &&
                          order.map((i) => (
                            <div
                              key={i}
                              className="mc-seg"
                              style={{
                                flexGrow: Math.max(d[i], 0.00001),
                                display: d[i] < 0.004 ? "none" : undefined,
                                background: fills[i],
                              }}
                              onPointerMove={(e) => tip.show(e, optionTip(g, i))}
                              onPointerLeave={tip.hide}
                            />
                          ))}
                      </div>
                    )}
                    {kind === "순서형" && <span className="mc-val">{num(meanOf(p, g, s.key))}</span>}
                    {kind === "이분형" && <span className="mc-val">{d ? pct(d[0], 1) : "–"}</span>}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
