import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Archive, ArrowUp, BarChart3, LoaderCircle, Moon, Sun } from "lucide-react";
import type { ChatMessage, ChatStatus, CompareKey, CompareWith, ContextItem, MessagePart, PanelState, Scenario } from "../types";
import type { Theme } from "../lib/color";
import { Markdown } from "../lib/markdown";
import { ContextCards } from "./ContextCards";
import { ExperimentalCard } from "./ExperimentalCard";
import { ResultCard } from "./ResultCard";
import { UnderstandingCard } from "./UnderstandingCard";
import { Hint } from "./Tooltip";

interface Props {
  messages: ChatMessage[];
  scenarios: Scenario[];
  status: ChatStatus | null;
  busy: boolean;
  theme: Theme;
  /** 상세 패널이 열려 있다 (채팅 열이 좁아진다) */
  split: boolean;
  /** 상세 패널이 보이고 있는 장면 */
  detailKey: string | null;
  /** 목록에 없는 새 문항도 예측할 수 있다 (실시간 모드) */
  canPredictNew: boolean;
  onToggleTheme: () => void;
  onSend: (text: string, scenario?: Scenario) => void;
  onDetail: (state: PanelState, compare: CompareKey | null) => void;
  onOpenReport: (item: ContextItem, pages: number[]) => void;
  /** 비교 보기: 평소 답변 옆에 함께 띄울 단계 */
  compareWith: CompareWith;
  onCompareWith: (v: CompareWith) => void;
}

/** 비교 보기의 칸 이름 */
const SIDE_LABEL: Record<"none" | "data" | "full", { name: string; note: string }> = {
  none: { name: "일반 AI", note: "질문만 받음 · 조사 데이터와 문서 없음" },
  data: { name: "RAG 없음", note: "조사 데이터 + 단어 검색" },
  full: { name: "수원서베이 AI", note: "조사 데이터 + 개념 사전 + 참고 문서" },
};

const COMPARE_OPTIONS: { value: CompareWith; label: string }[] = [
  { value: "off", label: "끔" },
  { value: "none", label: "일반 AI와" },
  { value: "data", label: "RAG 없이와" },
];

const keyOf = (s: PanelState) => `${s.qkey}|${s.attribute}|${s.adapter}`;

/** 화면의 중심인 대화 열. 답변 안에 결과 카드와 참고 문서가 들어간다. */
export function Chat({
  messages,
  scenarios,
  status,
  busy,
  theme,
  split,
  detailKey,
  canPredictNew,
  onToggleTheme,
  onSend,
  onDetail,
  onOpenReport,
  compareWith,
  onCompareWith,
}: Props) {
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const stick = useRef(true); // 맨 아래를 보고 있을 때만 새 글을 따라 내려간다
  const count = messages.length;

  // 새 질문을 보내면 맨 아래로 간다
  useEffect(() => {
    stick.current = true;
  }, [count]);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  });

  // 결과 카드가 자료를 받아 커질 때도 따라 내려간다
  useEffect(() => {
    const el = scrollRef.current;
    const inner = el?.firstElementChild;
    if (!el || !inner) return;
    const ro = new ResizeObserver(() => {
      if (stick.current) el.scrollTop = el.scrollHeight;
    });
    ro.observe(inner);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, [draft]);

  // 답변을 받는 중에 Enter를 치면 보내지 않는다. 조용히 무시하지 않고 이유를 알린다 (친 글은 그대로 둔다).
  const [blocked, setBlocked] = useState(false);
  useEffect(() => {
    if (!busy) setBlocked(false);
  }, [busy]);

  const submit = () => {
    const text = draft.trim();
    if (!text) return;
    if (busy) {
      setBlocked(true);
      return;
    }
    setDraft("");
    onSend(text);
  };

  const empty = count === 0;
  // 비교 답변이 하나라도 있거나 비교를 켜 두었으면 열을 넓힌다. 설정을 꺼도 지난 비교 답변은 두 칸 그대로 보인다
  const duo = compareWith !== "off" || messages.some((m) => m.compareAnswer);

  /** 답변 본문: 글, 결과 카드, 진행 표시, 안내 */
  /** compact: 차트 없이 한 줄로 줄인 카드. 비교 답변의 두 칸은 좁아서 늘 줄인다 */
  const body = (a: { parts: MessagePart[]; status?: string; pending?: boolean; notice?: string }, compact: boolean) => (
    <>
      {a.parts.map((part, i) =>
        part.kind === "text" ? (
          <Markdown key={i} text={part.text} />
        ) : part.kind === "experimental" ? (
          <ExperimentalCard key={i} p={part.prediction} theme={theme} />
        ) : (
          <ResultCard
            key={i}
            state={part.state}
            compare={part.compare}
            theme={theme}
            compact={compact}
            active={detailKey === keyOf(part.state)}
            onDetail={onDetail}
          />
        ),
      )}
      {a.status && <div className="msg-status">{a.status}…</div>}
      {a.pending && !a.status && !(a.parts.at(-1)?.kind === "text") && (
        <div className="typing" aria-label="답변을 쓰는 중">
          <i />
          <i />
          <i />
        </div>
      )}
      {a.notice && <div className="msg-notice">{a.notice}</div>}
    </>
  );

  return (
    <section className={`chat${split ? " is-split" : ""}`}>
      <header className="top">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            <BarChart3 size={15} />
          </span>
          <span className="brand-name">수원서베이 특화 LLM</span>
        </div>
        <button className="icon-btn" onClick={onToggleTheme} aria-label={theme === "light" ? "어두운 화면으로" : "밝은 화면으로"}>
          {theme === "light" ? <Moon size={17} /> : <Sun size={17} />}
        </button>
      </header>

      <div
        className="chat-scroll"
        ref={scrollRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        }}
      >
        <div className={`chat-col${empty ? " is-empty" : ""}${duo ? " is-duo" : ""}`}>
          {empty && (
            <div className="hero">
              <h1>수원서베이 특화 LLM</h1>
              <p>
                2023·2024년 수원서베이로 학습한 로컬 AI가 아직 하지 않은 조사의 응답을 미리 가늠합니다.
                <br />
                조사표 문구를 몰라도 평소 말로 물어보세요. 어떤 문항으로 이해했는지 함께 보여 드립니다.
              </p>
              {canPredictNew && <p className="hero-new">새 문항도 물어볼 수 있습니다 (실험적)</p>}
              <div className="intro-cards">
                {scenarios.map((s) => (
                  <button key={s.id} className="intro-card" disabled={busy} onClick={() => onSend(s.prompt, s)}>
                    <span className="intro-title">{s.title}</span>
                    <span className="intro-prompt">{s.prompt}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="msg msg-user">
                <div className="bubble">{m.content}</div>
              </div>
            ) : (
              <div key={m.id} className="msg msg-assistant">
                {m.saved && (
                  <span
                    className="saved-badge"
                    title={`대화 모델이 지금 쓴 답이 아니라 미리 저장해 둔 답변이다 (${m.saved.model}, ${m.saved.saved_at.slice(0, 10)})`}
                  >
                    <Archive size={12} aria-hidden /> 저장된 답변
                  </span>
                )}
                {m.understanding && (
                  <UnderstandingCard
                    u={m.understanding}
                    picked={m.parts.flatMap((p) => (p.kind === "card" ? [p.state.qkey] : []))}
                  />
                )}
                {m.compareAnswer ? (
                  <div className="duo">
                    <div className="duo-col is-side">
                      <DuoLabel grounding={m.compareAnswer.grounding} />
                      {body(m.compareAnswer, true)}
                    </div>
                    <div className="duo-col is-main">
                      <DuoLabel grounding="full" />
                      {body(m, true)}
                    </div>
                  </div>
                ) : (
                  body(m, split)
                )}
                {m.context && m.context.length > 0 && !m.pending && <ContextCards items={m.context} onOpen={onOpenReport} />}
              </div>
            ),
          )}
        </div>
      </div>

      <footer className="chat-foot">
        <div className="chat-col">
          <div className="ground" role="radiogroup" aria-label="비교해서 보기">
            <span className="ground-label">비교해서 보기</span>
            {COMPARE_OPTIONS.map((o) => (
              <button
                key={o.value}
                role="radio"
                aria-checked={compareWith === o.value}
                className={`chip${compareWith === o.value ? " is-on" : ""}`}
                disabled={busy}
                onClick={() => onCompareWith(o.value)}
              >
                {o.label}
              </button>
            ))}
          </div>
          {!empty && (
            <div className="quick" aria-label="추천 질문">
              {scenarios.map((s) => (
                <Hint key={s.id} text={s.prompt}>
                  <button className="chip" disabled={busy} onClick={() => onSend(s.prompt, s)}>
                    {s.title}
                  </button>
                </Hint>
              ))}
            </div>
          )}
          <div className="composer">
            <textarea
              ref={inputRef}
              rows={1}
              value={draft}
              placeholder={busy ? "답변을 받는 중입니다…" : "예: 연령별로 버스 만족도를 보여 줘"}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                // 한글 조합 중의 Enter는 보내지 않는다
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  submit();
                }
              }}
              aria-label="질문 입력"
            />
            <button
              className={`send${busy ? " is-busy" : ""}`}
              onClick={submit}
              disabled={busy || !draft.trim()}
              aria-label={busy ? "답변을 받는 중" : "보내기"}
              title={busy ? "답변을 받는 중입니다" : undefined}
            >
              {busy ? <LoaderCircle size={17} className="spin" /> : <ArrowUp size={17} />}
            </button>
          </div>
          {busy && blocked && <div className="chat-status">답변을 받는 중입니다. 끝나면 보낼 수 있습니다.</div>}
          {status && !status.available && (
            <div className="chat-status">대화 모델이 연결되지 않았습니다. 추천 질문의 결과는 볼 수 있습니다.</div>
          )}
        </div>
      </footer>
    </section>
  );
}

function DuoLabel({ grounding }: { grounding: "none" | "data" | "full" }) {
  const l = SIDE_LABEL[grounding];
  return (
    <div className={`duo-label${grounding === "full" ? " is-main" : ""}`}>
      <strong>{l.name}</strong>
      <span>{l.note}</span>
    </div>
  );
}
