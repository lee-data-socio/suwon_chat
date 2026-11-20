import { MessageSquare, PanelLeftClose, PanelLeftOpen, SquarePen, Trash2 } from "lucide-react";
import type { SessionSummary } from "../types";

/** 지난 시각을 짧게 */
function ago(seconds: number): string {
  const d = Math.max(0, Date.now() / 1000 - seconds);
  if (d < 60) return "방금";
  if (d < 3600) return `${Math.floor(d / 60)}분 전`;
  if (d < 86400) return `${Math.floor(d / 3600)}시간 전`;
  return `${Math.floor(d / 86400)}일 전`;
}

/** 왼쪽 대화 기록. 서버 메모리에만 있어 서버를 끄면 사라진다. */
export function Sidebar({
  sessions,
  activeId,
  open,
  busy,
  onToggle,
  onNew,
  onOpen,
  onDelete,
}: {
  sessions: SessionSummary[];
  activeId: string | null;
  open: boolean;
  busy: boolean;
  onToggle: () => void;
  onNew: () => void;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  if (!open)
    return (
      <nav className="side is-closed" aria-label="대화 기록">
        <button className="icon-btn" onClick={onToggle} aria-label="대화 기록 열기" title="대화 기록 열기">
          <PanelLeftOpen size={17} />
        </button>
        <button className="icon-btn" onClick={onNew} disabled={busy} aria-label="새 대화" title="새 대화">
          <SquarePen size={17} />
        </button>
      </nav>
    );
  return (
    <nav className="side" aria-label="대화 기록">
      <div className="side-top">
        <button className="icon-btn" onClick={onToggle} aria-label="대화 기록 닫기" title="대화 기록 닫기">
          <PanelLeftClose size={17} />
        </button>
        <button className="side-new" onClick={onNew} disabled={busy}>
          <SquarePen size={15} aria-hidden /> 새 대화
        </button>
      </div>
      <div className="side-head">대화 기록</div>
      <div className="side-list">
        {sessions.length === 0 ? (
          <p className="side-empty">아직 대화가 없습니다. 질문을 보내면 여기에 남습니다.</p>
        ) : (
          sessions.map((s) => (
            <div key={s.id} className={`side-item${s.id === activeId ? " is-active" : ""}`}>
              <button
                className="side-open"
                onClick={() => onOpen(s.id)}
                disabled={busy && s.id !== activeId}
                aria-current={s.id === activeId ? "true" : undefined}
                title={s.title}
              >
                <MessageSquare size={14} aria-hidden />
                <span className="side-title">{s.title}</span>
                <span className="side-meta">
                  {ago(s.updated_at)} · 질문 {s.count}개
                </span>
              </button>
              <button
                className="side-del"
                onClick={() => onDelete(s.id)}
                disabled={busy && s.id === activeId}
                aria-label={`'${s.title}' 대화 지우기`}
                title="지우기"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))
        )}
      </div>
      <p className="side-foot">대화 기록은 서버를 끄면 사라집니다.</p>
    </nav>
  );
}
