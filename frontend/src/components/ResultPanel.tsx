import { useState } from "react";
import { BookOpen, ChevronDown, Database, Radio, Table2, ZoomIn } from "lucide-react";
import type { CompareKey, Meta, PanelState, Prediction, QuestionSummary } from "../types";
import type { Theme } from "../lib/color";
import { hasLastYear, seriesFor } from "../lib/format";
import { adapterLabel, REFERENCE_ID } from "../lib/labels";
import { BinaryChart } from "./BinaryChart";
import { DataTable } from "./DataTable";
import { NominalChart } from "./NominalChart";
import { OrdinalChart } from "./OrdinalChart";
import { QuestionPicker } from "./QuestionPicker";
import { SeriesMark } from "./SeriesMark";
import { SummaryStrip } from "./SummaryStrip";
import { Hint } from "./Tooltip";

interface Props {
  meta: Meta;
  questions: QuestionSummary[];
  state: PanelState;
  prediction: Prediction | null;
  loading: boolean;
  error: string | null;
  compare: Set<CompareKey>;
  theme: Theme;
  onState: (patch: Partial<PanelState>) => void;
  onCompare: (key: CompareKey) => void;
}

export function ResultPanel({ meta, questions, state, prediction: p, loading, error, compare, theme, onState, onCompare }: Props) {
  const [picking, setPicking] = useState(false);
  const [table, setTable] = useState(false);
  const [zoom, setZoom] = useState(false);

  const live = (p?.source ?? meta.source) === "live";

  return (
    <main className="panel" aria-busy={loading}>
      <header className="panel-head">
        <div className="q-block">
          {p ? (
            <>
              <div className="q-meta">
                <span>{p.question.survey_no}</span>
                <span>
                  {p.question.scale_type}
                  {p.question.scale_type !== "이분형" && ` · 보기 ${p.question.options.length}개`}
                </span>
                <span>{p.predict_year}년 조사 문항</span>
                {p.question.held_out && (
                  <Hint className="badge badge-held" text="이 어댑터는 이 문항을 어느 연도에서도 보지 않고 학습했다.">
                    학습에서 뺀 문항
                  </Hint>
                )}
                {!p.question.scored && (
                  <Hint
                    className="badge badge-quiet"
                    text="보고서의 종합 오차는 3개 연도에 모두 있는 순서형 문항으로만 계산했다. 이 문항은 거기에 들어가지 않는다."
                  >
                    종합 오차 밖
                  </Hint>
                )}
              </div>
              <h1 className="q-text" title={p.question.text}>
                {p.question.item ? (
                  <>
                    <span className="q-stem">{p.question.stem}</span>
                    <span className="q-item">{p.question.item.replace(/^-\s*/, "")}</span>
                  </>
                ) : (
                  p.question.text
                )}
              </h1>
            </>
          ) : (
            <h1 className="q-text q-placeholder">{error ? "예측을 불러오지 못했습니다" : "불러오는 중"}</h1>
          )}
        </div>
        <div className="head-actions">
          <Hint
            className={`source-badge${live ? " is-live" : ""}`}
            text={
              live
                ? "모델을 지금 돌려 얻은 예측이다."
                : "미리 계산해 저장해 둔 예측이다. 모델을 지금 돌린 값이 아니다."
            }
          >
            {live ? <Radio size={14} aria-hidden /> : <Database size={14} aria-hidden />}
            {live ? "실시간" : "사전 계산"}
          </Hint>
          <button className="btn" onClick={() => setPicking(true)}>
            문항 바꾸기 <ChevronDown size={15} aria-hidden />
          </button>
        </div>
      </header>

      <div className="controls">
        <div className="control">
          <span className="control-label">집단 나누기</span>
          <div className="seg-group" role="radiogroup" aria-label="집단을 나누는 속성">
            {meta.attributes.map((a) => (
              <button
                key={a.id}
                role="radio"
                aria-checked={state.attribute === a.id}
                className={`seg-btn${state.attribute === a.id ? " is-on" : ""}`}
                onClick={() => onState({ attribute: a.id })}
              >
                {a.label}
              </button>
            ))}
          </div>
        </div>
        <div className="control">
          <span className="control-label">어댑터</span>
          <div className="seg-group" role="radiogroup" aria-label="예측에 쓰는 어댑터">
            {meta.adapters.map((a) => (
              <Hint
                key={a.id}
                text={
                  <>
                    {a.description}
                    {a.held_out_groups.length > 0 && (
                      <div className="tip-foot">
                        뺀 집단: {a.held_out_groups.map((g) => `${g.attribute_label} ${g.group}`).join(", ")}
                      </div>
                    )}
                  </>
                }
              >
                <button
                  role="radio"
                  aria-checked={state.adapter === a.id}
                  className={`seg-btn${state.adapter === a.id ? " is-on" : ""}`}
                  onClick={() => onState({ adapter: a.id })}
                >
                  {a.label}
                </button>
              </Hint>
            ))}
          </div>
        </div>
      </div>

      {error && !p && <div className="panel-error">{error}</div>}

      {p && (
        <div className={`panel-body${loading ? " is-loading" : ""}`}>
          <SummaryStrip p={p} />

          <section className="card chart-card">
            <div className="card-head">
              <h2>
                예측과 {p.predict_year}년 실측
                <span className="card-sub">
                  {p.attribute.label} {p.groups.length}개 집단 · {p.adapter.label} ({p.adapter.short})
                </span>
              </h2>
              <div className="card-tools">
                <span className="tools-label">함께 보기</span>
                <button
                  className={`chip${compare.has("last_year") && hasLastYear(p) ? " is-on" : ""}`}
                  disabled={!hasLastYear(p)}
                  title={hasLastYear(p) ? undefined : "작년 조사와 보기가 달라 견줄 수 없다"}
                  onClick={() => onCompare("last_year")}
                >
                  <SeriesMark series="last_year" size={9} /> 작년({p.predict_year - 1}) 실측
                </button>
                {p.adapter.id !== REFERENCE_ID && (
                  <button className={`chip${compare.has("reference") ? " is-on" : ""}`} onClick={() => onCompare("reference")}>
                    <SeriesMark series="reference" size={9} /> {adapterLabel(REFERENCE_ID)}
                  </button>
                )}
                <button className={`chip${compare.has("zero_shot") ? " is-on" : ""}`} onClick={() => onCompare("zero_shot")}>
                  <SeriesMark series="zero_shot" size={9} /> 학습 전 모델
                </button>
                <span className="tools-gap" />
                {p.question.scale_type === "순서형" && !table && (
                  <button
                    className={`chip${zoom ? " is-on" : ""}`}
                    onClick={() => setZoom((z) => !z)}
                    title="평균 비교의 가로축을 자료 범위로 좁힌다"
                  >
                    <ZoomIn size={13} aria-hidden /> 평균 확대
                  </button>
                )}
                <button className={`chip${table ? " is-on" : ""}`} onClick={() => setTable((t) => !t)}>
                  <Table2 size={13} aria-hidden /> 표로 보기
                </button>
              </div>
            </div>

            <div className="card-body">
              {(() => {
                const series = seriesFor(p, compare);
                if (table) return <DataTable p={p} series={series} />;
                if (p.question.scale_type === "순서형")
                  return <OrdinalChart p={p} series={series} theme={theme} zoom={zoom} />;
                if (p.question.scale_type === "이분형") return <BinaryChart p={p} series={series} />;
                return <NominalChart p={p} series={series} />;
              })()}
            </div>

            <div className="card-foot">
              {seriesFor(p, compare).map((s) => (
                <span className="legend-item" key={s.key}>
                  <SeriesMark series={s.key} size={10} bar={!table && p.question.scale_type !== "순서형"} />
                  {s.name}
                </span>
              ))}
            </div>
          </section>

          <section className="context" aria-label="참고 맥락">
            <BookOpen size={16} aria-hidden />
            <div>
              <strong>
                <span className="layer-no">③</span> 참고 맥락
              </strong>
              <span className="context-rule">예측에 사용되지 않은 참고 자료</span>
              <p>참고 문서는 아직 연결되지 않았습니다. 연결하면 이 문항과 관련된 보고서 대목이 문서 이름·쪽수와 함께 여기에 나옵니다.</p>
            </div>
          </section>
        </div>
      )}

      {picking && (
        <QuestionPicker
          adapters={meta.adapters}
          questions={questions}
          current={state.qkey}
          onClose={() => setPicking(false)}
          onPick={(qkey) => {
            setPicking(false);
            onState({ qkey });
          }}
        />
      )}
    </main>
  );
}
