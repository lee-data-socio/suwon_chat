import { Check, ChevronDown, Search } from "lucide-react";
import type { QuestionCandidate, Understanding } from "../types";

/**
 * 질문 이해 칸. 사용자의 질문을 그대로 넣어 단어 검색과 개념 사전 검색(RAG)이 찾은 문항 후보를 나란히 보인다.
 * 대화 모델이 실제로 고른 문항(답변의 결과 카드)은 표시한다.
 */
export function UnderstandingCard({ u, picked }: { u: Understanding; picked: string[] }) {
  const chosen = new Set(picked);
  return (
    <details className="und" open>
      <summary className="und-head">
        <Search size={14} aria-hidden />
        <strong>질문 이해</strong>
        <span>같은 질문으로 찾은 문항 후보</span>
        <ChevronDown size={14} className="und-chevron" aria-hidden />
      </summary>
      <div className="und-cols">
        <Column
          title="단어 검색"
          tag="RAG 없음"
          note="질문과 글자가 겹치는 문항"
          items={u.words}
          chosen={chosen}
          empty="글자가 겹치는 문항이 없습니다"
        />
        <Column
          title="개념 사전 검색"
          tag="RAG"
          note="조사 개념의 정의와 일상 표현으로 찾은 문항"
          items={u.concepts}
          chosen={chosen}
          empty={u.concepts_available ? "찾은 문항이 없습니다" : "검색 색인을 쓸 수 없습니다"}
          rag
        />
      </div>
    </details>
  );
}

function Column({
  title,
  tag,
  note,
  items,
  chosen,
  empty,
  rag = false,
}: {
  title: string;
  tag: string;
  note: string;
  items: QuestionCandidate[];
  chosen: Set<string>;
  empty: string;
  rag?: boolean;
}) {
  return (
    <div className={`und-col${rag ? " is-rag" : ""}`}>
      <div className="und-title">
        <span>{title}</span>
        <em>{tag}</em>
      </div>
      <p className="und-note">{note}</p>
      {items.length ? (
        <ol className="und-list">
          {items.map((it) => {
            const on = chosen.has(it.qkey);
            return (
              <li key={it.qkey} className={on ? "is-picked" : undefined}>
                <span className="und-text">{it.text}</span>
                {it.concept && <span className="und-concept">{it.concept}</span>}
                {on && (
                  <span className="und-picked">
                    <Check size={12} aria-hidden /> 답변에 쓴 문항
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      ) : (
        <p className="und-empty">{empty}</p>
      )}
    </div>
  );
}
