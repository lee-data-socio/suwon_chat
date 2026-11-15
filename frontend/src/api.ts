import type { ChatStatus, CompareKey, Meta, PanelState, Prediction, QuestionSummary, Scenario } from "./types";

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

const cache = new Map<string, Prediction>();
const key = (s: PanelState) => `${s.qkey}|${s.attribute}|${s.adapter}`;

export function cachedPrediction(s: PanelState): Prediction | undefined {
  return cache.get(key(s));
}

export function storePrediction(s: PanelState, p: Prediction) {
  cache.set(key(s), p);
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
  | { type: "done"; stop_reason: string }
  | { type: "error"; code: string; message: string };

/** 대화를 보내고 SSE를 읽는다. POST라서 EventSource 대신 fetch 스트림을 쓴다. */
export async function streamChat(
  messages: { role: "user" | "assistant"; content: string }[],
  panel: PanelState | null,
  onEvent: (e: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, panel }),
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
