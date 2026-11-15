import { useEffect, useRef, useState } from "react";
import { ArrowUp, BarChart3, Moon, Sun } from "lucide-react";
import type { ChatMessage, ChatStatus, Scenario } from "../types";
import type { Theme } from "../lib/color";
import { Markdown } from "../lib/markdown";
import { Hint } from "./Tooltip";

interface Props {
  messages: ChatMessage[];
  scenarios: Scenario[];
  status: ChatStatus | null;
  busy: boolean;
  theme: Theme;
  onToggleTheme: () => void;
  onSend: (text: string, scenario?: Scenario) => void;
  onRevisit: (m: ChatMessage) => void;
}

export function ChatPane({ messages, scenarios, status, busy, theme, onToggleTheme, onSend, onRevisit }: Props) {
  const [draft, setDraft] = useState("");
  const listRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }, [draft]);

  const submit = () => {
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    onSend(text);
  };

  const empty = messages.length === 0;

  return (
    <aside className="chat">
      <header className="chat-head">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            <BarChart3 size={17} />
          </span>
          <div>
            <div className="brand-name">수원서베이 응답 예측</div>
            <div className="brand-sub">조사 전에 응답 분포를 미리 그려 보는 시연</div>
          </div>
        </div>
        <button className="icon-btn" onClick={onToggleTheme} aria-label={theme === "light" ? "어두운 화면으로" : "밝은 화면으로"}>
          {theme === "light" ? <Moon size={17} /> : <Sun size={17} />}
        </button>
      </header>

      <div className="chat-list" ref={listRef}>
        {empty && (
          <div className="chat-intro">
            <p className="intro-lead">문항과 집단을 말하면 예측과 실측을 오른쪽에 띄웁니다.</p>
            <p className="intro-sub">
              2023·2024년 수원서베이로 학습한 모델이 2025년 조사 결과를 보지 않고 예측한 응답 분포입니다. 아래 질문으로 시작해 보세요.
            </p>
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
        {messages.map((m) => (
          <div key={m.id} className={`msg msg-${m.role}`}>
            {m.role === "user" ? (
              <div className="bubble">{m.content}</div>
            ) : (
              <div className="answer">
                {m.status && <div className="msg-status">{m.status}…</div>}
                {m.content && <Markdown text={m.content} />}
                {m.pending && !m.content && !m.status && (
                  <div className="typing" aria-label="답변을 쓰는 중">
                    <i />
                    <i />
                    <i />
                  </div>
                )}
                {m.notice && <div className="msg-notice">{m.notice}</div>}
                {m.panel && !m.pending && (
                  <button className="revisit" onClick={() => onRevisit(m)}>
                    <BarChart3 size={13} aria-hidden /> 이 결과 다시 보기
                  </button>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <footer className="chat-foot">
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
            placeholder="예: 연령별로 버스 만족도를 보여 줘"
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
          <button className="send" onClick={submit} disabled={busy || !draft.trim()} aria-label="보내기">
            <ArrowUp size={17} />
          </button>
        </div>
        <div className={`chat-status${status?.available ? " is-on" : ""}`}>
          <i aria-hidden />
          {status == null
            ? "대화 모델을 확인하는 중"
            : status.available
              ? "대화 모델 연결됨"
              : "대화 모델이 연결되지 않았습니다 · 패널은 그대로 쓸 수 있습니다"}
        </div>
      </footer>
    </aside>
  );
}
