// 앱 서버 응답의 모양. backend/suwon_chat/service.py 와 api.py 가 기준이다.

export type ScaleType = "순서형" | "이분형" | "명목형";
export type Source = "precomputed" | "live";

export interface AdapterInfo {
  id: string;
  label: string;
  short: string;
  description: string;
  held_out_groups: { attribute: string; attribute_label: string; group: string }[];
  held_out_questions: string[];
}

export interface AttributeInfo {
  id: string;
  label: string;
  groups: { group: string; n: number | null }[];
}

export interface Meta {
  predict_year: number;
  source: Source;
  /** 집단·문항 제외 어댑터를 견줄 기준 어댑터의 id */
  reference_adapter: string;
  adapters: AdapterInfo[];
  attributes: AttributeInfo[];
}

export interface QuestionSummary {
  qkey: string;
  text: string;
  stem: string;
  item: string;
  survey_no: string;
  scale_type: ScaleType;
  options: string[];
  three_year: boolean;
  error: number;
  error_metric: "wd" | "tv";
  held_out_in: string[];
}

export interface PredictionQuestion extends QuestionSummary {
  ordinal: number[];
  option_values: (number | null)[];
  scale_min: number | null;
  scale_max: number | null;
  scored: boolean;
  held_out: boolean;
  history_years: number[];
}

export interface HistoryRow {
  year: number;
  dist: number[];
  mean: number | null;
  error: number;
}

export interface GroupRow {
  group: string;
  n: number | null;
  /** 응답자가 100명 이하 */
  small_sample: boolean;
  held_out: boolean;
  predicted: number[];
  actual: number[];
  zero_shot: number[];
  predicted_mean: number | null;
  actual_mean: number | null;
  zero_shot_mean: number | null;
  error: number;
  zero_shot_error: number;
  last_year_error: number | null;
  lower_bound: number | null;
  history: HistoryRow[];
  reference: null | { adapter: string; label: string; predicted: number[]; predicted_mean: number | null; error: number };
}

export interface Summary {
  error_metric: "wd" | "tv";
  error: number;
  error_readable: { unit: "points" | "percent"; value: number; scale_points: number | null; text: string };
  zero_shot_error: number;
  last_year_error: number | null;
  lower_bound: number | null;
  reference_error: number | null;
  order_corr: number | null;
  last_year_order_corr: number | null;
  spread_predicted: number | null;
  spread_actual: number | null;
}

export interface Prediction {
  source: Source;
  predict_year: number;
  adapter: { id: string; label: string; short: string; description: string };
  attribute: { id: string; label: string };
  question: PredictionQuestion;
  groups: GroupRow[];
  summary: Summary;
}

export interface PanelState {
  qkey: string;
  attribute: string;
  adapter: string;
}

/** 예측·실측 말고 함께 볼 수 있는 것 */
export type CompareKey = "last_year" | "reference" | "zero_shot";

export interface Scenario {
  id: string;
  title: string;
  prompt: string;
  panel: null | (PanelState & { compare: "actual" | "last_year" | "reference" });
}

export interface ChatStatus {
  available: boolean;
  model: string;
  reason: string | null;
}

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  /** 답변이 오는 중 */
  pending?: boolean;
  /** 도구 진행 표시 ("예측을 불러오는 중") */
  status?: string;
  /** 답변 대신 보이는 안내 (연결 안 됨, 오류) */
  notice?: string;
  /** 이 답변이 패널에 띄운 상태. 누르면 다시 볼 수 있다 */
  panel?: PanelState;
  compare?: CompareKey[];
}
