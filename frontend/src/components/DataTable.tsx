import type { Prediction } from "../types";
import type { SeriesDef } from "../lib/format";
import { count, distOf, errorOf, meanOf, num } from "../lib/format";
import { SeriesMark } from "./SeriesMark";

/** 차트와 같은 값을 표로 본다. 차트에서 숫자를 읽기 어려울 때 쓴다. */
export function DataTable({ p, series }: { p: Prediction; series: SeriesDef[] }) {
  const q = p.question;
  const hasMean = q.scale_type === "순서형";
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>집단</th>
            <th>값</th>
            {q.options.map((o, i) => (
              <th key={i} className="num" title={o}>
                {o}
              </th>
            ))}
            {hasMean && <th className="num">평균</th>}
            <th className="num">오차</th>
          </tr>
        </thead>
        <tbody>
          {p.groups.map((g) =>
            series.map((s, si) => {
              const d = distOf(p, g, s.key);
              return (
                <tr key={`${g.group}-${s.key}`} className={si === 0 ? "first" : ""}>
                  {si === 0 && (
                    <th rowSpan={series.length} scope="rowgroup">
                      {g.group}
                      <span className="table-n">{count(g.n)}</span>
                      {g.small_sample && <span className="badge badge-warn">표본 적음</span>}
                      {g.held_out && <span className="badge badge-held">학습에서 뺌</span>}
                    </th>
                  )}
                  <td className="series-cell">
                    <SeriesMark series={s.key} size={9} />
                    {s.name}
                  </td>
                  {q.options.map((_, i) => (
                    <td key={i} className="num">
                      {d ? (d[i] * 100).toFixed(1) : "–"}
                    </td>
                  ))}
                  {hasMean && <td className="num">{num(meanOf(p, g, s.key))}</td>}
                  <td className="num">{s.key === "actual" ? "" : num(errorOf(g, s.key), 3)}</td>
                </tr>
              );
            }),
          )}
        </tbody>
      </table>
      <p className="table-note">보기 칸의 값은 응답 비율(%)이다. 무응답은 빼고 계산했다.</p>
    </div>
  );
}
