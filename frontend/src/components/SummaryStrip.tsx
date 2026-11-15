import type { Prediction } from "../types";
import { cardStats, corr, metricName, num, readable } from "../lib/format";
import { adapterLabel, REFERENCE_ID } from "../lib/labels";
import { Hint } from "./Tooltip";

/** 오차 요약. 좋은 숫자만 고르지 않고, 견줄 대상과 집단 차이의 약점까지 한 줄에 둔다. */
export function SummaryStrip({ p }: { p: Prediction }) {
  const s = p.summary;
  const q = p.question;
  // 오차를 읽는 문구는 앱 서버가 만든 것을 그대로 쓴다. 숫자 부분만 크게 보이게 나눈다.
  const text = s.error_readable.text;
  const m = text.match(/^(.*?)(약\s*[\d.]+\s*(?:%p|%|점))(.*)$/);
  const [heroLead, heroValue, heroTrail] = m ? [m[1].trim(), m[2], m[3].trim()] : ["", text, ""];
  const refName = adapterLabel(REFERENCE_ID);

  // 학습에서 뺀 집단이 있으면 대화 속 카드는 그 집단의 오차를 앞세운다. 여기 큰 숫자는 집단 평균이므로,
  // 같은 문구로 그 집단의 오차를 바로 아래에 적어 두 숫자가 무엇인지 헷갈리지 않게 한다.
  const held = p.groups.filter((g) => g.held_out);
  const stats = held.length === 1 ? cardStats(p, "reference") : null;
  const heldLine = stats
    ? `${stats.main.label} ${readable(p, held[0].error)}` + (stats.versus ? ` · ${stats.versus.label} ${stats.versus.value}` : "")
    : q.held_out
      ? "이 어댑터가 학습에서 뺀 문항이다"
      : null;

  const rows: { key: string; label: string; hint: string; value: number | null }[] = [
    {
      key: "predicted",
      label: `${p.adapter.label}의 예측`,
      hint: "지금 차트에 나온 예측의 오차. 집단별 오차의 평균이다.",
      value: s.error,
    },
    {
      key: "reference",
      label: `${refName}의 예측`,
      hint: `같은 집단·문항을 보고 학습한 ${refName}의 오차. 빼고 학습한 어댑터와 견준다.`,
      value: s.reference_error,
    },
    {
      key: "last_year",
      label: `작년(${p.predict_year - 1}) 실측을 그대로 쓰면`,
      hint: "모델을 쓰지 않고 작년 조사 결과를 올해 예측으로 그대로 썼을 때의 오차.",
      value: s.last_year_error,
    },
    {
      key: "zero_shot",
      label: "학습 전 모델",
      hint: "어댑터를 붙이지 않은, 수원서베이로 학습하지 않은 원래 언어 모델의 오차.",
      value: s.zero_shot_error,
    },
    {
      key: "lower",
      label: "사람 응답의 표본오차",
      hint: "같은 크기로 조사를 다시 해도 생기는 차이. 어떤 예측도 이보다 작아지기 어렵다.",
      value: s.lower_bound,
    },
  ].filter((x) => x.value != null);
  const top = Math.max(...rows.map((x) => x.value as number), 0.0001);

  const unitText = (v: number | null) =>
    v == null ? "–" : q.scale_type === "순서형" ? `${v.toFixed(2)}점` : `${(v * 100).toFixed(1)}%p`;
  const spreadTop = Math.max(s.spread_predicted ?? 0, s.spread_actual ?? 0, 0.0001);
  // 실제 집단 차이가 아주 작으면(순서형 0.1점, 이분형 3%p 미만) 높낮이 순서는 우연에 가깝다.
  // 집단이 3개뿐이어도 상관은 쉽게 ±1이 된다. 두 경우 모두 상관을 확신처럼 보이지 않게 한다.
  const tinySpread = s.spread_actual != null && s.spread_actual < (q.scale_type === "순서형" ? 0.1 : 0.03);
  const weakOrder = tinySpread || p.groups.length <= 3;

  return (
    <section className="summary" aria-label="오차 요약">
      <div className="tile tile-hero">
        <div className="tile-label">
          <Hint text={`${p.attribute.label} ${p.groups.length}개 집단의 예측이 ${p.predict_year}년 실측과 평균적으로 얼마나 다른지 나타낸다.`}>
            집단 평균 오차 <span className="tile-lead">{p.groups.length}개 집단의 평균</span>
          </Hint>
        </div>
        <div className={`hero${m ? "" : " is-text"}`}>
          {heroLead && <span className="hero-word">{heroLead} </span>}
          <span className="hero-value">{heroValue}</span>
          {heroTrail && <span className="hero-word">{heroTrail}</span>}
        </div>
        <div className="tile-sub">
          {/* 뺀 집단의 줄이 길어 자리가 모자라면 분포 거리는 옆의 '오차 견주기' 첫 줄에서 본다 */}
          {!stats && (
            <span>
              {metricName(s.error_metric)} {num(s.error, 3)}
            </span>
          )}
          {heldLine && (
            <span className={`held-line${stats ? " is-wrap" : ""}`} title={heldLine}>
              {heldLine}
            </span>
          )}
        </div>
      </div>

      <div className="tile tile-ladder">
        <div className="tile-label">
          오차 견주기 <span className="tile-lead">{metricName(s.error_metric)} · 짧을수록 실측에 가깝다</span>
        </div>
        <div className="ladder">
          {rows.map((x) => (
            <Hint className="ladder-row" key={x.key} text={x.hint}>
              <span className="ladder-name">{x.label}</span>
              <span className="ladder-plot">
                <i className={`ladder-bar ladder-${x.key}`} style={{ width: `${((x.value as number) / top) * 100}%` }} />
              </span>
              <span className="ladder-value">{num(x.value, 3)}</span>
            </Hint>
          ))}
        </div>
      </div>

      <div className="tile tile-groups">
        <div className="tile-label">
          집단 차이 <span className="tile-lead">예측이 집단을 얼마나 구분하는가</span>
        </div>
        {s.spread_actual == null ? (
          <p className="tile-empty">명목형 문항은 평균이 없어 집단의 높낮이를 견주지 않는다.</p>
        ) : (
          <div className="groups-stats">
            <Hint
              className="stat"
              text="집단별 예측 평균과 실측 평균의 상관. +1에 가까우면 높낮이 순서가 실제와 같고, 0 근처나 음수이면 순서를 믿기 어렵다. 집단이 적거나 집단 차이가 작으면 우연히도 크게 나온다."
            >
              <span className="stat-name">높낮이 일치 · 집단 {p.groups.length}개</span>
              {s.order_corr == null ? (
                <span className="stat-note">집단이 2개라 계산하지 않는다</span>
              ) : (
                <>
                  <strong className={`stat-value${weakOrder ? " is-weak" : ""}`}>{corr(s.order_corr)}</strong>
                  <span className="stat-note">
                    {tinySpread
                      ? "집단 차이가 아주 작아 방향을 따지지 않는다"
                      : p.groups.length <= 3
                        ? "집단이 3개뿐이라 참고만 한다"
                        : corrWords(s.order_corr)}
                  </span>
                </>
              )}
            </Hint>
            <Hint
              className="stat"
              text="집단 평균의 최댓값과 최솟값의 차이. 예측의 벌어짐이 실측보다 작으면 집단 차이를 실제보다 작게 예측한 것이다."
            >
              <span className="stat-name">벌어짐</span>
              <span className="spread">
                <span className="spread-row">
                  <em>예측</em>
                  <span className="spread-plot">
                    <i className="spread-bar spread-pred" style={{ width: `${((s.spread_predicted ?? 0) / spreadTop) * 100}%` }} />
                  </span>
                  <b>{unitText(s.spread_predicted)}</b>
                </span>
                <span className="spread-row">
                  <em>실측</em>
                  <span className="spread-plot">
                    <i className="spread-bar spread-actual" style={{ width: `${((s.spread_actual ?? 0) / spreadTop) * 100}%` }} />
                  </span>
                  <b>{unitText(s.spread_actual)}</b>
                </span>
              </span>
            </Hint>
          </div>
        )}
      </div>
    </section>
  );
}

function corrWords(v: number): string {
  if (v >= 0.7) return "실제와 같은 방향";
  if (v >= 0.3) return "대체로 같은 방향";
  if (v > -0.3) return "관계가 거의 없다";
  return "실제와 반대 방향";
}
