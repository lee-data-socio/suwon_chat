import { useCallback, useEffect, useRef, useState } from "react";
import {
  cachedPrediction,
  fetchChatStatus,
  fetchMeta,
  fetchPrediction,
  fetchQuestions,
  fetchScenarios,
  storePrediction,
  streamChat,
} from "./api";
import type { ChatMessage, ChatStatus, CompareKey, Meta, PanelState, Prediction, QuestionSummary, Scenario } from "./types";
import { useTheme } from "./lib/hooks";
import { REFERENCE_ID, setAdapters } from "./lib/labels";
import { ChatPane } from "./components/ChatPane";
import { ResultPanel } from "./components/ResultPanel";
import { TipProvider } from "./components/Tooltip";

const FALLBACK: PanelState = { qkey: "q189_W2025", attribute: "TENURE", adapter: "exp1" };
const COMPARE_KEYS: CompareKey[] = ["last_year", "reference", "zero_shot"];

/** 주소의 ?q=&a=&m=&with= 로 패널 상태를 정할 수 있다. 발표 때 장면을 즐겨찾기로 열어 두는 데 쓴다. */
function fromUrl(): { panel: Partial<PanelState>; compare: CompareKey[] | null } {
  const u = new URLSearchParams(window.location.search);
  const panel: Partial<PanelState> = {};
  if (u.get("q")) panel.qkey = u.get("q")!;
  if (u.get("a")) panel.attribute = u.get("a")!;
  if (u.get("m")) panel.adapter = u.get("m")!;
  const withKeys = u.get("with");
  const compare = withKeys === null ? null : (withKeys.split(",").filter((k) => COMPARE_KEYS.includes(k as CompareKey)) as CompareKey[]);
  return { panel, compare };
}

const NO_CHAT = "대화 모델이 연결되지 않아 답변은 나오지 않습니다.";

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [meta, setMeta] = useState<Meta | null>(null);
  const [questions, setQuestions] = useState<QuestionSummary[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [chatStatus, setChatStatus] = useState<ChatStatus | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);

  const [panel, setPanel] = useState<PanelState>(FALLBACK);
  const [compare, setCompare] = useState<Set<CompareKey>>(new Set());
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [panelError, setPanelError] = useState<string | null>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const nextId = useRef(1);
  const panelRef = useRef(panel);
  panelRef.current = panel;
  const compareRef = useRef(compare);
  compareRef.current = compare;

  useEffect(() => {
    Promise.all([fetchMeta(), fetchQuestions(), fetchScenarios()])
      .then(([m, q, s]) => {
        setAdapters(m.adapters, m.reference_adapter);
        setMeta(m);
        setQuestions(q);
        setScenarios(s);
        const first = s.find((x) => x.panel)?.panel;
        const base = first ? { qkey: first.qkey, attribute: first.attribute, adapter: first.adapter } : FALLBACK;
        const url = fromUrl();
        setPanel({ ...base, ...url.panel });
        if (url.compare) setCompare(new Set(url.compare));
        else if (url.panel.adapter && url.panel.adapter !== REFERENCE_ID) setCompare(new Set(["reference"]));
      })
      .catch((e: Error) => setBootError(e.message));
    fetchChatStatus()
      .then(setChatStatus)
      .catch(() => setChatStatus({ available: false, model: "", reason: null }));
  }, []);

  // 패널 상태가 바뀌면 예측을 가져온다. 가져오는 동안에는 이전 차트를 흐리게 둔다.
  useEffect(() => {
    if (!meta) return;
    const hit = cachedPrediction(panel);
    if (hit) {
      setPrediction(hit);
      setPanelError(null);
      setLoading(false);
      return;
    }
    let alive = true;
    setLoading(true);
    fetchPrediction(panel)
      .then((p) => {
        if (!alive) return;
        setPrediction(p);
        setPanelError(null);
      })
      .catch((e: Error) => alive && setPanelError(`예측을 불러오지 못했습니다: ${e.message}`))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [panel, meta]);

  // 지금 장면을 주소에 적어 둔다 (새로 고치거나 즐겨찾기로 다시 열 수 있다)
  useEffect(() => {
    if (!meta) return;
    const u = new URLSearchParams({ q: panel.qkey, a: panel.attribute, m: panel.adapter });
    if (compare.size) u.set("with", [...compare].join(","));
    window.history.replaceState(null, "", `?${u}`);
  }, [panel, compare, meta]);

  /** 패널 상태를 바꾼다. 빼고 학습한 어댑터로 넘어가면 전체 학습 어댑터와 견주는 보기를 함께 켠다. */
  const applyPanel = useCallback((next: PanelState, nextCompare?: CompareKey[]) => {
    const prev = panelRef.current;
    setPanel(next);
    setCompare((c) => {
      if (nextCompare) return new Set(nextCompare);
      if (next.adapter !== REFERENCE_ID && prev.adapter === REFERENCE_ID) return new Set([...c, "reference"]);
      return c;
    });
  }, []);

  const patchPanel = useCallback(
    (patch: Partial<PanelState>) => applyPanel({ ...panelRef.current, ...patch }),
    [applyPanel],
  );

  const toggleCompare = useCallback((key: CompareKey) => {
    setCompare((c) => {
      const n = new Set(c);
      if (n.has(key)) n.delete(key);
      else n.add(key);
      return n;
    });
  }, []);

  const patchMessage = (id: number, fn: (m: ChatMessage) => ChatMessage) =>
    setMessages((list) => list.map((m) => (m.id === id ? fn(m) : m)));

  const send = useCallback(
    async (text: string, scenario?: Scenario) => {
      if (busy) return;
      const userMsg: ChatMessage = { id: nextId.current++, role: "user", content: text };
      const answerId = nextId.current++;

      // 추천 질문은 패널 상태가 정해져 있다. 대화 모델을 기다리지 않고 바로 띄운다.
      let shown: PanelState | undefined;
      let shownCompare: CompareKey[] | undefined;
      if (scenario?.panel) {
        const { compare: cmp, ...state } = scenario.panel;
        shown = state;
        shownCompare = cmp === "actual" ? [] : [cmp];
        applyPanel(state, shownCompare);
      }

      if (!chatStatus?.available) {
        setMessages((list) => [
          ...list,
          userMsg,
          {
            id: answerId,
            role: "assistant",
            content: "",
            notice: shown ? `${NO_CHAT} 오른쪽 패널은 이 질문에 맞춰 바꿨습니다.` : NO_CHAT,
            panel: shown,
            compare: shownCompare,
          },
        ]);
        return;
      }

      const history = [...messages, userMsg]
        .filter((m) => m.content.trim())
        .map((m) => ({ role: m.role, content: m.content }));
      setMessages((list) => [
        ...list,
        userMsg,
        { id: answerId, role: "assistant", content: "", pending: true, panel: shown, compare: shownCompare },
      ]);
      setBusy(true);
      try {
        await streamChat(history, shown ?? panelRef.current, (e) => {
          switch (e.type) {
            case "status":
              // 도구를 부르기 전에 쓴 글과 그 뒤에 쓰는 글이 붙지 않게 문단을 나눈다
              patchMessage(answerId, (m) => ({
                ...m,
                status: e.text,
                content: m.content && !m.content.endsWith("\n\n") ? `${m.content.trimEnd()}\n\n` : m.content,
              }));
              break;
            case "text":
              patchMessage(answerId, (m) => ({ ...m, status: undefined, content: m.content + e.delta }));
              break;
            case "panel":
              storePrediction(e.state, e.prediction);
              {
                // 대화 모델이 고른 비교 대상을 그대로 켠다. 앞 질문에서 켜 둔 것이 남지 않게 한다.
                const cmp: CompareKey[] = e.compare === "actual" ? [] : [e.compare];
                applyPanel(e.state, cmp);
                patchMessage(answerId, (m) => ({ ...m, panel: e.state, compare: cmp }));
              }
              break;
            case "done":
              patchMessage(answerId, (m) => ({ ...m, pending: false, status: undefined }));
              break;
            case "error":
              patchMessage(answerId, (m) => ({ ...m, pending: false, status: undefined, notice: e.message }));
              if (e.code === "no_api_key") setChatStatus((s) => (s ? { ...s, available: false } : s));
              break;
          }
        });
      } catch {
        patchMessage(answerId, (m) => ({ ...m, notice: "서버와 연결이 끊겼습니다. 패널은 그대로 쓸 수 있습니다." }));
      } finally {
        patchMessage(answerId, (m) => ({ ...m, pending: false, status: undefined }));
        setBusy(false);
      }
    },
    [busy, chatStatus, messages, applyPanel],
  );

  const revisit = useCallback(
    (m: ChatMessage) => m.panel && applyPanel(m.panel, m.compare ?? [...compareRef.current]),
    [applyPanel],
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
      <div className="app">
        <ChatPane
          messages={messages}
          scenarios={scenarios}
          status={chatStatus}
          busy={busy}
          theme={theme}
          onToggleTheme={toggleTheme}
          onSend={send}
          onRevisit={revisit}
        />
        {meta ? (
          <ResultPanel
            meta={meta}
            questions={questions}
            state={panel}
            prediction={prediction}
            loading={loading}
            error={panelError}
            compare={compare}
            theme={theme}
            onState={patchPanel}
            onCompare={toggleCompare}
          />
        ) : (
          <main className="panel panel-boot">불러오는 중</main>
        )}
      </div>
    </TipProvider>
  );
}
