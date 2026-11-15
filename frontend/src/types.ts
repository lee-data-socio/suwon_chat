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
  /** 목록에 없는 새 문항도 예측할 수 있는가 (실시간 모드) */
  can_predict_new?: boolean;
  /** 실시간 모드에서 돌리고 있는 모델 */
  live?: LiveInfo | null;
  adapters: AdapterInfo[];
  attributes: AttributeInfo[];
}

export interface LiveInfo {
  base_model: string;
  device: string;
  dtype: string;
  four_bit: boolean;
  adapters: string[];
  load_seconds: number;
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
  /** 실측이 없는 새 문항의 예측. 검증되지 않았다 */
  experimental?: boolean;
  /** 실험적 예측에 붙는 설명 */
  note?: string;
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

/** 참고 문서에서 찾은 대목 하나. 예측에 쓰이지 않은 자료다. */
export interface ContextItem {
  doc_id: string;
  title: string;
  page: number;
  pages: number;
  /** 문서 전체의 한두 문장 요약 */
  summary?: string;
  /** 찾아낸 쪽의 대목 */
  text: string;
  score: number;
  /** 시연용으로 만든 가상 문서 */
  synthetic: boolean;
}

export interface ContextResult {
  available: boolean;
  query: string;
  items: ContextItem[];
}

export interface Report {
  doc_id: string;
  title: string;
  summary: string;
  synthetic: boolean;
  pages: { page: number; text: string }[];
}

/** 답변은 글과 결과 카드가 도착한 순서대로 놓인다. */
export type MessagePart =
  | { kind: "text"; text: string }
  | { kind: "card"; state: PanelState; compare: CompareKey | null }
  /** 목록에 없는 새 문항의 예측. 실측과 오차가 없다 */
  | { kind: "experimental"; prediction: Prediction };

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  /** 글만 이은 것. 다음 질문 때 대화 모델에 넘긴다 */
  content: string;
  parts: MessagePart[];
  /** 답변이 오는 중 */
  pending?: boolean;
  /** 도구 진행 표시 ("예측을 불러오는 중") */
  status?: string;
  /** 답변 대신 보이는 안내 (연결 안 됨, 오류) */
  notice?: string;
  /** 이 답변에 붙은 참고 문서 */
  context?: ContextItem[];
  /** 대화 모델이 지금 쓴 답이 아니라 미리 저장해 둔 답변일 때 */
  saved?: { saved_at: string; model: string };
}
