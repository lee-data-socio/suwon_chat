import { useMemo } from "react";
import type { GroupRow, Prediction } from "../types";
import type { Theme } from "../lib/color";
import { ordinalSwatches, stackOrder } from "../lib/color";
import type { SeriesDef } from "../lib/format";
import { distOf, errorOf, meanOf, num, optionWords, pct } from "../lib/format";
import { useSize } from "../lib/hooks";
import { isMain, lineSizes } from "../lib/layout";
import { GroupLabel } from "./GroupLabel";
import { SeriesMark } from "./SeriesMark";
import { Hint, useTip } from "./Tooltip";

interface Props {
  p: Prediction;
  series: SeriesDef[];
  theme: Theme;
  zoom: boolean;
}

const TAG_W = 56;

/** 순서형 문항: 집단마다 ① 응답 분포(100% 누적 막대)와 ② 척도 평균을 한 줄에 나란히 놓는다. */
export function OrdinalChart({ p, series, theme, zoom }: Props) {
  const q = p.question;
  const swatches = useMemo(() => ordinalSwatches(q.option_values, theme), [q.option_values, theme]);
  const order = useMemo(() => stackOrder(q.option_values), [q.option_values]);
  const [rowsRef, rowsSize] = useSize<HTMLDivElement>();
  const [barRef, barSize] = useSize<HTMLDivElement>();
  const tip = useTip();

  // 막대 굵기: 줄 높이에 맞춘다. 예측·실측은 굵게, 함께 보는 것은 가늘게.
  const rowH = p.groups.length ? rowsSize.height / p.groups.length : 0;
  const sizes = lineSizes(rowH, series, rowH > 150 ? 26 : 22);
  const barW = Math.max(0, barSize.width - TAG_W);

  const scaleMin = q.scale_min ?? 0;
  const scaleMax = q.scale_max ?? 1;
  const domain = useMemo(() => {
    if (!zoom) return [scaleMin, scaleMax] as const;
    const means = p.groups.flatMap((g) => series.map((s) => meanOf(p, g, s.key))).filter((v): v is number => v != null);
    if (!means.length) return [scaleMin, scaleMax] as const;
    const lo = Math.min(...means);
    const hi = Math.max(...means);
    const pad = Math.max(0.15, (hi - lo) * 0.25);
    return [Math.max(scaleMin, Math.floor((lo - pad) * 10) / 10), Math.min(scaleMax, Math.ceil((hi + pad) * 10) / 10)] as const;
  }, [zoom, p, series, scaleMin, scaleMax]);
  const ticks = useMemo(() => makeTicks(domain[0], domain[1], zoom), [domain, zoom]);
  const pos = (v: number) => ((v - domain[0]) / (domain[1] - domain[0])) * 100;

  const lowWord = optionWords(q.options[order[0]]);
  const highIdx = [...order].reverse().find((i) => q.option_values[i] !== null) ?? order[order.length - 1];
  const highWord = optionWords(q.options[highIdx]);

  const segTip = (g: GroupRow, i: number) => (
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

  const meanTip = (g: GroupRow) => (
    <div>
      <div className="tip-title">{g.group} · 척도 평균</div>
      {series.map((s) => {
        const m = meanOf(p, g, s.key);
        return (
          m != null && (
            <div className="tip-row" key={s.key}>
              <SeriesMark series={s.key} size={9} />
              <span className="tip-name">{s.name}</span>
              <strong>{num(m)}</strong>
            </div>
          )
        );
      })}
      <div className="tip-foot">
        {num(scaleMin, 0)} = {lowWord} · {num(scaleMax, 0)} = {highWord}
      </div>
    </div>
  );

  return (
    <div className={`chart ordinal${series.length >= 4 ? " wide-err" : ""}`}>
      <div className="chart-head chart-grid">
        <div className="col-title">집단</div>
        <div className="col-head">
          <div className="col-title">
            <span className="layer-no">①</span> 응답 분포
            <span className="col-note">막대 안 숫자는 응답 비율(%)</span>
          </div>
          <ScaleLegend p={p} swatches={swatches} order={order} />
        </div>
        <div className="col-head">
          <div className="col-title">
            <span className="layer-no">②</span> 척도 평균
            {zoom ? (
              <span className="badge badge-warn">확대됨 · 척도 전체가 아닙니다</span>
            ) : (
              <span className="col-note">가로축은 척도 전체 ({num(scaleMin, 0)}~{num(scaleMax, 0)})</span>
            )}
          </div>
          <div className="axis">
            <div className="axis-ends">
              <span title={q.options[order[0]]}>← {lowWord}</span>
              <span title={q.options[highIdx]}>{highWord} →</span>
            </div>
            <div className="axis-ticks">
              {ticks.map((t) => (
                <span key={t} style={{ left: `${pos(t)}%` }}>
                  {Number.isInteger(t) ? t : t.toFixed(1)}
                </span>
              ))}
            </div>
          </div>
        </div>
        <div className="col-title col-right">
          <Hint text="2025년 실측 분포와의 거리(WD). 0이면 두 분포가 같고, 클수록 다르다.">오차</Hint>
        </div>
      </div>

      <div className="chart-rows" ref={rowsRef}>
        {p.groups.map((g, gi) => (
          <div className={`chart-row chart-grid${g.held_out ? " is-held" : ""}`} key={g.group}>
            <GroupLabel g={g} />
            <div className="bars" ref={gi === 0 ? barRef : undefined}>
              {series.map((s, si) => {
                const d = distOf(p, g, s.key);
                const { line, bar: h, font } = sizes[si];
                return (
                  <div className={`bar-line${isMain(s.key) ? "" : " is-minor"}`} key={s.key} style={{ height: line }}>
                    <span className="bar-tag" style={{ width: TAG_W, fontSize: font }}>
                      {s.tag}
                    </span>
                    <div className="stack" style={{ height: h }} role="img" aria-label={`${g.group} ${s.name}`}>
                      {d &&
                        order.map((i) => {
                          const v = d[i];
                          const w = v * barW;
                          return (
                            <div
                              key={i}
                              className="seg"
                              style={{
                                flexGrow: Math.max(v, 0.00001),
                                display: v < 0.0005 ? "none" : undefined,
                                background: swatches[i].fill,
                                color: swatches[i].ink,
                              }}
                              onPointerMove={(e) => tip.show(e, segTip(g, i))}
                              onPointerLeave={tip.hide}
                            >
                              {h >= 13 && w >= 30 && <span>{Math.round(v * 100)}</span>}
                            </div>
                          );
                        })}
                    </div>
                  </div>
                );
              })}
            </div>
            <div
              className="mean-track"
              tabIndex={0}
              onPointerMove={(e) => tip.show(e, meanTip(g))}
              onPointerLeave={tip.hide}
              onFocus={(e) => tip.show(e.currentTarget, meanTip(g))}
              onBlur={tip.hide}
            >
              <div className="mean-plot">
                <div className="mean-line" />
                {ticks.map((t) => (
                  <i className="mean-tick" key={t} style={{ left: `${pos(t)}%` }} />
                ))}
                <Connector a={g.predicted_mean} b={g.actual_mean} pos={pos} />
                {[...series].reverse().map((s) => {
                  const m = meanOf(p, g, s.key);
                  if (m == null) return null;
                  const x = pos(m);
                  if (x < -1 || x > 101) return null;
                  return (
                    <span className="mean-dot" key={s.key} style={{ left: `${x}%` }}>
                      <SeriesMark series={s.key} size={isMain(s.key) ? 13 : 10} />
                    </span>
                  );
                })}
              </div>
              <div className="mean-values">
                <span>
                  <SeriesMark series="predicted" size={8} />
                  {num(g.predicted_mean)}
                </span>
                <span>
                  <SeriesMark series="actual" size={8} />
                  {num(g.actual_mean)}
                </span>
              </div>
            </div>
            <ErrorCell g={g} series={series} />
          </div>
        ))}
      </div>
    </div>
  );
}

function Connector({ a, b, pos }: { a: number | null; b: number | null; pos: (v: number) => number }) {
  if (a == null || b == null) return null;
  const x1 = Math.max(0, Math.min(100, pos(Math.min(a, b))));
  const x2 = Math.max(0, Math.min(100, pos(Math.max(a, b))));
  return <div className="mean-link" style={{ left: `${x1}%`, width: `${x2 - x1}%` }} />;
}

export function ErrorCell({ g, series }: { g: GroupRow; series: SeriesDef[] }) {
  return (
    <div className="err-cell">
      <div className="err-main">{num(g.error, 3)}</div>
      <div className="err-subs">
        {series
          .filter((s) => s.key !== "predicted" && s.key !== "actual")
          .map((s) => (
            <span className="err-sub" key={s.key}>
              {s.tag} {num(errorOf(g, s.key), 3)}
            </span>
          ))}
      </div>
    </div>
  );
}

function makeTicks(lo: number, hi: number, zoom: boolean): number[] {
  const span = hi - lo;
  if (!zoom) {
    const all: number[] = [];
    for (let v = Math.ceil(lo); v <= hi + 1e-9; v++) all.push(v);
    if (all.length <= 8) return all;
    return [lo, (lo + hi) / 2, hi];
  }
  const step = [0.1, 0.2, 0.25, 0.5, 1, 2].find((s) => span / s <= 6) ?? 2;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
  return out;
}

/** 보기 색의 범례. 보기가 적으면 모두 적고, 많으면 양 끝과 가운데만 적는다. */
export function ScaleLegend({
  p,
  swatches,
  order,
}: {
  p: Prediction;
  swatches: { fill: string }[];
  order: number[];
}) {
  const q = p.question;
  const valid = order.filter((i) => q.option_values[i] !== null);
  const off = order.filter((i) => q.option_values[i] === null);
  const labelAll = valid.length <= 5 && valid.reduce((n, i) => n + q.options[i].length, 0) <= 34;
  const mid = valid.length % 2 === 1 && valid.length >= 5 ? valid[(valid.length - 1) / 2] : null;
  return (
    <div className="scale-legend">
      {labelAll ? (
        valid.map((i) => (
          <span className="legend-item" key={i}>
            <i style={{ background: swatches[i].fill }} />
            {q.options[i]}
          </span>
        ))
      ) : (
        <>
          <span className="legend-end" title={q.options[valid[0]]}>
            {q.options[valid[0]]}
          </span>
          <span className="legend-ramp">
            {valid.map((i) => (
              <i key={i} style={{ background: swatches[i].fill }} title={q.options[i]} />
            ))}
          </span>
          <span className="legend-end" title={q.options[valid[valid.length - 1]]}>
            {q.options[valid[valid.length - 1]]}
          </span>
          {mid !== null && optionWords(q.options[mid]) !== q.options[mid] && (
            <span className="legend-item legend-mid">
              <i style={{ background: swatches[mid].fill }} />
              {optionWords(q.options[mid])}
            </span>
          )}
        </>
      )}
      {off.map((i) => (
        <span className="legend-item" key={i}>
          <i style={{ background: swatches[i].fill }} />
          {q.options[i]}
        </span>
      ))}
    </div>
  );
}
