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
  CompareWith,
  ContextItem,
  Meta,
  MessagePart,
  PanelState,
  QuestionSummary,
  Scenario,
  SideAnswer,
} from "./types";
import { useTheme } from "./lib/hooks";
import { REFERENCE_ID, setAdapters, setLive } from "./lib/labels";
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

/** 답변 조각 끝에 글을 잇는다. 카드 뒤에 새로 시작하는 글은 앞의 빈 줄을 뗀다 */
function appendText(parts: MessagePart[], delta: string): MessagePart[] {
  const next = parts.slice();
  const last = next[next.length - 1];
  if (last?.kind === "text") next[next.length - 1] = { kind: "text", text: last.text + delta };
  else {
    const t = delta.replace(/^\s+/, "");
    if (t) next.push({ kind: "text", text: t });
  }
  return next;
}

const COMPARE_STORE = "suwon-compare-with";

function loadCompareWith(): CompareWith {
  try {
    const v = localStorage.getItem(COMPARE_STORE);
    return v === "none" || v === "data" ? v : "off";
  } catch {
    return "off";
  }
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
  // 비교 보기: 같은 질문을 근거를 달리해(일반 AI / RAG 없음) 한 번 더 받아 평소 답변 옆에 띄운다
  const [compareWith, setCompareWith] = useState<CompareWith>(loadCompareWith);
  useEffect(() => {
    try {
      localStorage.setItem(COMPARE_STORE, compareWith);
    } catch {
      /* 저장하지 못해도 이번 화면에서는 그대로 쓴다 */
    }
  }, [compareWith]);
  const [busy, setBusy] = useState(false);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [report, setReport] = useState<{ item: ContextItem; pages: number[] } | null>(null);
  const nextId = useRef(1);
  const detailRef = useRef(detail);
  detailRef.current = detail;

  const { data: prediction, error: panelError, loading } = usePrediction(detail?.state ?? null);

  useEffect(() => {
    Promise.all([fetchMeta(), fetchQuestions(), fetchScenarios()])
      .then(([m, q, s]) => {
        setAdapters(m.adapters, m.reference_adapter);
        setLive(m.live, m.source === "live");
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
      const side = compareWith === "off" ? null : compareWith;
      setMessages((list) => [
        ...list,
        userMsg,
        {
          id: answerId,
          role: "assistant",
          content: "",
          parts,
          pending: true,
          ...(side ? { compareAnswer: { grounding: side, content: "", parts: [], pending: true } } : {}),
        },
      ]);
      setBusy(true);

      // 옆 칸: 자기만의 대화 이력으로 따로 받는다. 평소 답변(실제 수치)이 넘어가면 비교가 흐려진다.
      const patchSide = (fn: (s: SideAnswer) => SideAnswer) =>
        patchMessage(answerId, (m) => (m.compareAnswer ? { ...m, compareAnswer: fn(m.compareAnswer) } : m));
      const runSide = async () => {
        if (!side) return;
        type Turn = { role: "user" | "assistant"; content: string };
        const sideHistory = [...messages, userMsg].flatMap((m): Turn[] =>
          m.role === "user"
            ? [{ role: "user", content: m.content }]
            : m.compareAnswer?.content.trim()
              ? [{ role: "assistant", content: m.compareAnswer.content }]
              : [],
        );
        let ended = false;
        try {
          await streamChat(
            sideHistory,
            current,
            (e) => {
              switch (e.type) {
                case "status":
                  patchSide((s) => ({ ...s, status: e.text }));
                  break;
                case "text":
                  patchSide((s) => ({ ...s, parts: appendText(s.parts, e.delta), status: undefined, content: s.content + e.delta }));
                  break;
                case "panel": {
                  storePrediction(e.state, e.prediction);
                  const card: MessagePart = { kind: "card", state: e.state, compare: e.compare === "actual" ? null : e.compare };
                  patchSide((s) => ({ ...s, parts: [...s.parts, card], status: undefined }));
                  break;
                }
                case "new_prediction":
                  patchSide((s) => ({ ...s, parts: [...s.parts, { kind: "experimental", prediction: e.prediction }], status: undefined }));
                  break;
                case "done":
                  ended = true;
                  break;
                case "error":
                  ended = true;
                  patchSide((s) => ({ ...s, notice: e.message }));
                  break;
              }
            },
            undefined,
            undefined,
            side,
          );
          if (!ended) patchSide((s) => ({ ...s, notice: "답변이 중간에 끊겼습니다." }));
        } catch {
          patchSide((s) => ({ ...s, notice: "서버와 연결이 끊겼습니다." }));
        } finally {
          patchSide((s) => ({ ...s, pending: false, status: undefined }));
        }
      };
      const sideDone = runSide();
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
            case "new_prediction":
              // 목록에 없는 새 문항의 예측. 실측이 없으므로 평소 카드와 다른 실험적 카드로 놓는다.
              patchMessage(answerId, (m) => ({
                ...m,
                parts: [...m.parts, { kind: "experimental", prediction: e.prediction }],
                status: undefined,
              }));
              break;
            case "context":
              gotContext = true;
              // 한 답변에 여러 번 올 수 있다 (문항에 붙은 문서, 대화 모델이 따로 찾은 문서). 먼저 온 것을 앞에 두고 합친다.
              patchMessage(answerId, (m) => {
                const have = m.context ?? [];
                const seen = new Set(have.map((it) => `${it.doc_id}:${it.page}`));
                return { ...m, context: [...have, ...e.items.filter((it) => !seen.has(`${it.doc_id}:${it.page}`))] };
              });
              break;
            case "understanding": {
              const { type: _t, ...u } = e;
              patchMessage(answerId, (m) => ({ ...m, understanding: u }));
              break;
            }
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
        patchMessage(answerId, (m) => ({
          ...m,
          notice: m.parts.some((p) => p.kind === "card")
            ? "서버와 연결이 끊겼습니다. 결과 카드는 그대로 볼 수 있습니다."
            : "서버와 연결이 끊겼습니다. 잠시 뒤 다시 물어 주세요.",
        }));
      } finally {
        patchMessage(answerId, (m) => ({ ...m, pending: false, status: undefined }));
        await sideDone;
        setBusy(false);
      }
    },
    [busy, chatStatus, compareWith, messages, openDetail],
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
          canPredictNew={meta?.can_predict_new === true}
          onToggleTheme={toggleTheme}
          onSend={send}
          compareWith={compareWith}
          onCompareWith={setCompareWith}
          onDetail={openDetail}
          onOpenReport={(item, pages) => setReport({ item, pages })}
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
      {report && <ReportModal item={report.item} pages={report.pages} onClose={() => setReport(null)} />}
    </TipProvider>
  );
}
