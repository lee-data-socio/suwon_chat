import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchChatStatus,
  fetchContext,
  fetchMeta,
  fetchQuestions,
  fetchScenarios,
  storePrediction,
  streamChat,
} from "./api";
import type {
  ChatMessage,
  ChatStatus,
  CompareKey,
  ContextItem,
  Meta,
  MessagePart,
  PanelState,
  QuestionSummary,
  Scenario,
} from "./types";
import { useTheme } from "./lib/hooks";
import { REFERENCE_ID, setAdapters } from "./lib/labels";
import { usePrediction } from "./lib/usePrediction";
import { Chat } from "./components/Chat";
import { ReportModal } from "./components/ReportModal";
import { ResultPanel } from "./components/ResultPanel";
import { TipProvider } from "./components/Tooltip";

const FALLBACK: PanelState = { qkey: "q189_W2025", attribute: "TENURE", adapter: "exp1" };
const COMPARE_KEYS: CompareKey[] = ["last_year", "reference", "zero_shot"];

/** 상세 패널이 보이고 있는 장면 */
interface Detail {
  state: PanelState;
  compare: Set<CompareKey>;
}

const keyOf = (s: PanelState) => `${s.qkey}|${s.attribute}|${s.adapter}`;

/** 주소의 ?q=&a=&m=&with= 로 장면을 열 수 있다. 발표 때 즐겨찾기로 열어 두는 데 쓴다. */
function fromUrl(): { panel: Partial<PanelState>; compare: CompareKey[] | null } | null {
  const u = new URLSearchParams(window.location.search);
  if (!u.get("q")) return null;
  const panel: Partial<PanelState> = { qkey: u.get("q")! };
  if (u.get("a")) panel.attribute = u.get("a")!;
  if (u.get("m")) panel.adapter = u.get("m")!;
  const withKeys = u.get("with");
  const compare =
    withKeys === null ? null : (withKeys.split(",").filter((k) => COMPARE_KEYS.includes(k as CompareKey)) as CompareKey[]);
  return { panel, compare };
}

/** 대화에서 마지막으로 보여 준 장면. 다음 질문에서 "속성만 바꿔 줘" 같은 말을 풀 때 대화 모델에 알려 준다. */
function lastShown(messages: ChatMessage[]): PanelState | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const cards = messages[i].parts.filter((p) => p.kind === "card");
    const last = cards[cards.length - 1];
    if (last && last.kind === "card") return last.state;
  }
  return null;
}

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [meta, setMeta] = useState<Meta | null>(null);
  const [questions, setQuestions] = useState<QuestionSummary[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [chatStatus, setChatStatus] = useState<ChatStatus | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [report, setReport] = useState<ContextItem | null>(null);
  const nextId = useRef(1);
  const detailRef = useRef(detail);
  detailRef.current = detail;

  const { data: prediction, error: panelError, loading } = usePrediction(detail?.state ?? null);

  useEffect(() => {
    Promise.all([fetchMeta(), fetchQuestions(), fetchScenarios()])
      .then(([m, q, s]) => {
        setAdapters(m.adapters, m.reference_adapter);
        setMeta(m);
        setQuestions(q);
        setScenarios(s);
        const url = fromUrl();
        if (url) {
          // 주소로 열면 상세 패널이 열린 채로 시작한다
          const state = { ...FALLBACK, ...url.panel };
          const compare = url.compare ?? (state.adapter !== m.reference_adapter ? ["reference" as const] : []);
          setDetail({ state, compare: new Set(compare) });
        }
      })
      .catch((e: Error) => setBootError(e.message));
    fetchChatStatus()
      .then(setChatStatus)
      .catch(() => setChatStatus({ available: false, model: "", reason: null }));
  }, []);

  // 상세 패널의 장면을 주소에 적어 둔다. 패널을 닫으면 주소도 비운다.
  useEffect(() => {
    if (!meta) return;
    if (!detail) {
      window.history.replaceState(null, "", window.location.pathname);
      return;
    }
    const u = new URLSearchParams({ q: detail.state.qkey, a: detail.state.attribute, m: detail.state.adapter });
    if (detail.compare.size) u.set("with", [...detail.compare].join(","));
    window.history.replaceState(null, "", `?${u}`);
  }, [detail, meta]);

  const openDetail = useCallback((state: PanelState, compare: CompareKey | null) => {
    setDetail({ state, compare: new Set(compare ? [compare] : []) });
  }, []);

  /** 상세 패널 안에서 문항·속성·어댑터를 바꾼다. 대화 속 카드는 그대로 둔다. */
  const patchDetail = useCallback((patch: Partial<PanelState>) => {
    setDetail((d) => {
      if (!d) return d;
      const state = { ...d.state, ...patch };
      // 빼고 학습한 어댑터로 넘어가면 전체 학습 어댑터와 견주는 보기를 함께 켠다
      const compare =
        state.adapter !== REFERENCE_ID && d.state.adapter === REFERENCE_ID ? new Set([...d.compare, "reference" as const]) : d.compare;
      return { state, compare };
    });
  }, []);

  const toggleCompare = useCallback((key: CompareKey) => {
    setDetail((d) => {
      if (!d) return d;
      const compare = new Set(d.compare);
      if (compare.has(key)) compare.delete(key);
      else compare.add(key);
      return { ...d, compare };
    });
  }, []);

  const patchMessage = (id: number, fn: (m: ChatMessage) => ChatMessage) =>
    setMessages((list) => list.map((m) => (m.id === id ? fn(m) : m)));

  const send = useCallback(
    async (text: string, scenario?: Scenario) => {
      if (busy) return;
      const userMsg: ChatMessage = { id: nextId.current++, role: "user", content: text, parts: [] };
      const answerId = nextId.current++;

      // 추천 질문은 보여 줄 장면이 정해져 있다. 대화 모델을 기다리지 않고 결과 카드를 바로 놓는다.
      const parts: MessagePart[] = [];
      let shown: PanelState | null = null;
      if (scenario?.panel) {
        const { compare, ...state } = scenario.panel;
        const cmp = compare === "actual" ? null : compare;
        shown = state;
        parts.push({ kind: "card", state, compare: cmp });
        if (detailRef.current) openDetail(state, cmp); // 상세 패널이 열려 있으면 새 결과를 따라간다
      }

      // 대화 모델이 연결되지 않았어도 일단 보낸다. 추천 질문은 서버가 저장해 둔 답변으로 답할 수 있다.
      const history = [...messages, userMsg]
        .filter((m) => m.content.trim())
        .map((m) => ({ role: m.role, content: m.content }));
      const current = shown ?? detailRef.current?.state ?? lastShown(messages);
      setMessages((list) => [...list, userMsg, { id: answerId, role: "assistant", content: "", parts, pending: true }]);
      setBusy(true);
      let finished = false; // done이나 error를 받았는가
      let gotContext = false;
      try {
        await streamChat(
          history,
          current,
          (e) => {
          switch (e.type) {
            case "status":
              patchMessage(answerId, (m) => ({ ...m, status: e.text }));
              break;
            case "text":
              patchMessage(answerId, (m) => {
                const next = m.parts.slice();
                const last = next[next.length - 1];
                if (last?.kind === "text") next[next.length - 1] = { kind: "text", text: last.text + e.delta };
                else {
                  // 카드 뒤에 새로 시작하는 글은 앞의 빈 줄을 뗀다
                  const t = e.delta.replace(/^\s+/, "");
                  if (t) next.push({ kind: "text", text: t });
                }
                return { ...m, parts: next, status: undefined, content: m.content + e.delta };
              });
              break;
            case "panel": {
              storePrediction(e.state, e.prediction);
              const cmp = e.compare === "actual" ? null : e.compare;
              patchMessage(answerId, (m) => {
                const next = m.parts.slice();
                const at = next.findIndex((p) => p.kind === "card" && keyOf(p.state) === keyOf(e.state));
                const card: MessagePart = { kind: "card", state: e.state, compare: cmp };
                if (at >= 0) next[at] = card; // 추천 질문이 먼저 놓은 카드와 같은 장면이면 새로 놓지 않는다
                else next.push(card);
                return { ...m, parts: next, status: undefined };
              });
              if (detailRef.current) openDetail(e.state, cmp);
              break;
            }
            case "context":
              gotContext = true;
              patchMessage(answerId, (m) => ({ ...m, context: e.items }));
              break;
            case "saved":
              patchMessage(answerId, (m) => ({ ...m, saved: { saved_at: e.saved_at, model: e.model } }));
              break;
            case "done":
              finished = true;
              patchMessage(answerId, (m) => ({ ...m, pending: false, status: undefined }));
              break;
            case "error":
              finished = true;
              patchMessage(answerId, (m) => ({ ...m, pending: false, status: undefined, notice: e.message }));
              if (e.code === "no_api_key") setChatStatus((s) => (s ? { ...s, available: false } : s));
              // 설명은 못 받았어도 결과 카드는 이미 놓여 있다. 참고 문서는 화면이 직접 채운다.
              if (shown && !gotContext)
                void fetchContext(shown.qkey).then((items) => patchMessage(answerId, (m) => ({ ...m, context: items })));
              break;
          }
          },
          scenario?.id,
        );
        // 서버가 끝을 알리지 않고 닫았으면 잘린 답이다. 완성된 답처럼 두지 않는다.
        if (!finished) patchMessage(answerId, (m) => ({ ...m, notice: "답변이 중간에 끊겼습니다." }));
      } catch {
        patchMessage(answerId, (m) => ({ ...m, notice: "서버와 연결이 끊겼습니다. 결과 카드는 그대로 볼 수 있습니다." }));
      } finally {
        patchMessage(answerId, (m) => ({ ...m, pending: false, status: undefined }));
        setBusy(false);
      }
    },
    [busy, chatStatus, messages, openDetail],
  );

  if (bootError)
    return (
      <div className="boot-error">
        <strong>앱 서버에 연결하지 못했습니다.</strong>
        <p>{bootError}</p>
        <p>앱 서버(8000번)가 켜져 있는지 확인한 뒤 새로 고쳐 주세요.</p>
      </div>
    );

  return (
    <TipProvider>
      <div className={`app${detail ? " has-detail" : ""}`}>
        <Chat
          messages={messages}
          scenarios={scenarios}
          status={chatStatus}
          busy={busy}
          theme={theme}
          split={detail !== null}
          detailKey={detail ? keyOf(detail.state) : null}
          onToggleTheme={toggleTheme}
          onSend={send}
          onDetail={openDetail}
          onOpenReport={setReport}
        />
        {detail && meta && (
          <ResultPanel
            meta={meta}
            questions={questions}
            state={detail.state}
            prediction={prediction}
            loading={loading}
            error={panelError}
            compare={detail.compare}
            theme={theme}
            onState={patchDetail}
            onCompare={toggleCompare}
            onClose={() => setDetail(null)}
          />
        )}
      </div>
      {report && <ReportModal item={report} onClose={() => setReport(null)} />}
    </TipProvider>
  );
}
