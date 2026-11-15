import type {
  ChatStatus,
  CompareKey,
  ContextItem,
  ContextResult,
  Meta,
  PanelState,
  Prediction,
  QuestionSummary,
  Report,
  Scenario,
} from "./types";

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* 본문이 JSON이 아니면 상태 코드만 쓴다 */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const fetchMeta = () => get<Meta>("/api/meta");
export const fetchQuestions = () => get<QuestionSummary[]>("/api/questions");
export const fetchScenarios = () => get<Scenario[]>("/api/scenarios");
export const fetchChatStatus = () => get<ChatStatus>("/api/chat/status");
export const fetchReport = (docId: string) => get<Report>(`/api/reports/${encodeURIComponent(docId)}`);

/** 문항과 관련된 참고 문서. 참고 문서가 연결되지 않았거나 실패하면 빈 목록으로 본다 (그 영역을 그리지 않는다). */
export async function fetchContext(qkey: string): Promise<ContextItem[]> {
  try {
    const r = await get<ContextResult>(`/api/context?qkey=${encodeURIComponent(qkey)}`);
    return r.available ? r.items : [];
  } catch {
    return [];
  }
}

const cache = new Map<string, Prediction>();
const key = (s: PanelState) => `${s.qkey}|${s.attribute}|${s.adapter}`;

export function cachedPrediction(s: PanelState): Prediction | undefined {
  return cache.get(key(s));
}

/** 대화가 보낸 예측을 저장한다. 같은 장면을 기다리고 있는 카드가 있으면 바로 알려 준다 (실시간 모드에서는 모델을 두 번 기다리지 않는다). */
export function storePrediction(s: PanelState, p: Prediction) {
  cache.set(key(s), p);
  for (const fn of listeners) fn(key(s), p);
}

const listeners = new Set<(key: string, p: Prediction) => void>();

export function onPrediction(fn: (key: string, p: Prediction) => void): () => void {
  listeners.add(fn);
  return () => void listeners.delete(fn);
}

export async function fetchPrediction(s: PanelState): Promise<Prediction> {
  const hit = cache.get(key(s));
  if (hit) return hit;
  const q = new URLSearchParams({ qkey: s.qkey, attribute: s.attribute, adapter: s.adapter });
  const p = await get<Prediction>(`/api/prediction?${q}`);
  cache.set(key(s), p);
  return p;
}

export type ChatEvent =
  | { type: "status"; tool: string; text: string }
  | { type: "text"; delta: string }
  | { type: "panel"; state: PanelState; compare: "actual" | CompareKey; prediction: Prediction }
  | { type: "context"; qkey: string | null; query: string; items: ContextItem[] }
  | { type: "saved"; saved_at: string; model: string }
  | { type: "new_prediction"; prediction: Prediction }
  | { type: "done"; stop_reason: string }
  | { type: "error"; code: string; message: string };

/** 대화를 보내고 SSE를 읽는다. POST라서 EventSource 대신 fetch 스트림을 쓴다. */
export async function streamChat(
  messages: { role: "user" | "assistant"; content: string }[],
  panel: PanelState | null,
  onEvent: (e: ChatEvent) => void,
  /** 추천 질문 버튼으로 보낸 경우 그 시나리오의 id. 대화 모델을 쓸 수 없을 때 서버가 저장된 답변을 찾는 데 쓴다 */
  scenarioId?: string,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(scenarioId ? { messages, panel, scenario_id: scenarioId } : { messages, panel }),
    signal,
  });
  if (!res.ok || !res.body) {
    onEvent({ type: "error", code: "http", message: `서버가 응답하지 않았습니다 (${res.status}).` });
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const flush = (block: string) => {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
    if (!data.length) return;
    try {
      onEvent({ type: event, ...JSON.parse(data.join("\n")) } as ChatEvent);
    } catch {
      /* 깨진 조각은 버린다 */
    }
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      flush(buffer.slice(0, cut));
      buffer = buffer.slice(cut + 2);
    }
  }
  if (buffer.trim()) flush(buffer);
}
