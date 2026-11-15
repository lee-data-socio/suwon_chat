import { useEffect, useMemo, useRef, useState } from "react";
import { Search, X } from "lucide-react";
import type { AdapterInfo, QuestionSummary, ScaleType } from "../types";
import { adapterLabel, REFERENCE_ID } from "../lib/labels";
import { num } from "../lib/format";

interface Props {
  adapters: AdapterInfo[];
  questions: QuestionSummary[];
  current: string;
  onPick: (qkey: string) => void;
  onClose: () => void;
}

const TYPES: ("전체" | ScaleType)[] = ["전체", "순서형", "이분형", "명목형"];

/** 문항 고르기. 113개 문항을 문항 줄기로 묶어 보이고 글자로 거른다. */
export function QuestionPicker({ adapters, questions, current, onPick, onClose }: Props) {
  const [query, setQuery] = useState("");
  const [type, setType] = useState<(typeof TYPES)[number]>("전체");
  const inputRef = useRef<HTMLInputElement | null>(null);
  const currentRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    inputRef.current?.focus();
    currentRef.current?.scrollIntoView({ block: "center" });
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const groups = useMemo(() => {
    const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
    const hits = questions.filter((q) => {
      if (type !== "전체" && q.scale_type !== type) return false;
      const hay = `${q.text} ${q.options.join(" ")} ${q.survey_no}`.toLowerCase();
      return tokens.every((t) => hay.includes(t));
    });
    const map = new Map<string, QuestionSummary[]>();
    for (const q of hits) {
      const k = `${q.survey_no}|${q.stem}`;
      if (!map.has(k)) map.set(k, []);
      map.get(k)!.push(q);
    }
    return { list: [...map.values()], total: hits.length };
  }, [questions, query, type]);

  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="picker" role="dialog" aria-modal="true" aria-label="문항 고르기">
        <div className="picker-head">
          <div className="picker-search">
            <Search size={16} aria-hidden />
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="문항에 들어간 말로 찾기 (예: 버스, 복지, 만족)"
              aria-label="문항 검색"
            />
          </div>
          <div className="picker-types" role="group" aria-label="문항 종류">
            {TYPES.map((t) => (
              <button key={t} className={`chip${type === t ? " is-on" : ""}`} onClick={() => setType(t)}>
                {t}
              </button>
            ))}
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="닫기">
            <X size={18} />
          </button>
        </div>
        <div className="picker-count">
          2025년 문항 {questions.length}개 가운데 {groups.total}개 · 오른쪽 숫자는 {adapterLabel(REFERENCE_ID)}의 평균 오차(작을수록 잘 맞음)
        </div>
        <div className="picker-list">
          {groups.list.length === 0 && <p className="picker-empty">맞는 문항이 없습니다.</p>}
          {groups.list.map((items) => {
            const head = items[0];
            const single = items.length === 1 && !head.item;
            return (
              <div className="picker-group" key={`${head.survey_no}|${head.stem}`}>
                {!single && (
                  <div className="picker-stem">
                    <span className="picker-no">{head.survey_no}</span>
                    {head.stem}
                  </div>
                )}
                {items.map((q) => (
                  <button
                    key={q.qkey}
                    ref={q.qkey === current ? currentRef : undefined}
                    className={`picker-item${q.qkey === current ? " is-current" : ""}${single ? " is-single" : ""}`}
                    onClick={() => onPick(q.qkey)}
                  >
                    <span className="picker-text">
                      {single && <span className="picker-no">{q.survey_no}</span>}
                      {single ? q.stem : q.item || q.text}
                    </span>
                    {q.held_out_in.length > 0 && (
                      <span className="badge badge-held">
                        {adapters.find((a) => a.id === q.held_out_in[0])?.label ?? "어댑터"}에서 뺀 문항
                      </span>
                    )}
                    <span className="picker-type">
                      {q.scale_type}
                      {q.scale_type !== "이분형" && ` ${q.options.length}`}
                    </span>
                    <span className="picker-error">{num(q.error, 3)}</span>
                  </button>
                ))}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
