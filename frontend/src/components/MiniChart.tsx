import { useEffect, useMemo, useState } from "react";
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
 * 요소의 배치 너비. 막대는 처음에 0에서 자라나는 애니메이션(transform)을 하므로,
 * 화면에 보이는 크기(getBoundingClientRect)가 아니라 배치 크기를 잰다.
 */
function useLayoutWidth() {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!el) return;
    setWidth(el.offsetWidth);
    const ro = new ResizeObserver(() => setWidth(el.offsetWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, width] as const;
}

/** 색 위에 올리는 글자색: 밝은 색(노랑, 밝은 회색) 위에는 검정, 나머지는 흰색 */
function inkOn(fill: string): string {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(fill.slice(i, i + 2), 16) / 255);
  const lin = (v: number) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
  const luminance = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  return luminance > 0.36 ? "#0b0b0b" : "#ffffff";
}

/**
 * 답변 안에 들어가는 작은 차트. 집단마다 예측 막대와 실측 막대를 위아래로 붙여,
 * 두 막대가 얼마나 겹치는지만 읽히게 한다. 자세한 숫자는 막대에 마우스를 올리거나 '자세히 보기'에서 본다.
 */
export function MiniChart({ p, series, theme }: Props) {
  const q = p.question;
  const kind = q.scale_type;
  const tip = useTip();
  const [stackRef, stackWidth] = useLayoutWidth(); // 막대 하나의 너비. 칸에 숫자가 들어갈지 정하는 데 쓴다
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
        {p.groups.map((g, gi) => (
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
              {series.map((s, si) => {
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
                      <div
                        className="mc-stack"
                        role="img"
                        aria-label={`${g.group} ${s.name}`}
                        ref={gi === 0 && si === 0 ? stackRef : undefined}
                      >
                        {d &&
                          order.map((i) => (
                            <div
                              key={i}
                              className="mc-seg"
                              style={{
                                flexGrow: Math.max(d[i], 0.00001),
                                display: d[i] < 0.004 ? "none" : undefined,
                                background: fills[i],
                                color: inkOn(fills[i]),
                              }}
                              onPointerMove={(e) => tip.show(e, optionTip(g, i))}
                              onPointerLeave={tip.hide}
                            >
                              {/* 명목형은 색만으로 보기 8개를 가리기 어렵다. 칸이 넓으면 비율을 적는다 */}
                              {kind === "명목형" && !minor && d[i] * stackWidth >= 34 && <span>{Math.round(d[i] * 100)}%</span>}
                            </div>
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
