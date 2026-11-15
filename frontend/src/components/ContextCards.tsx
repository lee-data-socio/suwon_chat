import { BookOpen } from "lucide-react";
import type { ContextItem } from "../types";
import { cleanTitle, plainSnippet } from "../lib/format";

/** 참고 문서 카드. 예측에 쓰이지 않은 자료라는 것을 머리말로 늘 밝힌다. */
export function ContextCards({ items, onOpen }: { items: ContextItem[]; onOpen: (item: ContextItem) => void }) {
  // 같은 문서의 여러 쪽이 걸리면 카드 하나로 묶는다 (요약이 같아 카드가 겹쳐 보인다). 누르면 가장 먼저 걸린 쪽으로 간다.
  const docs: { item: ContextItem; pages: number[] }[] = [];
  for (const it of items) {
    const hit = docs.find((d) => d.item.doc_id === it.doc_id);
    if (hit) hit.pages.push(it.page);
    else docs.push({ item: it, pages: [it.page] });
  }
  const shown = docs.slice(0, 3);
  if (!shown.length) return null;
  return (
    <section className="ctx" aria-label="참고 문서">
      <div className="ctx-head">
        <BookOpen size={14} aria-hidden />
        <strong>참고 문서</strong>
        <span>예측에 사용되지 않은 자료</span>
      </div>
      <div className="ctx-list">
        {shown.map(({ item: it, pages }) => (
          <button className="ctx-card" key={it.doc_id} onClick={() => onOpen(it)}>
            <span className="ctx-title">{cleanTitle(it.title)}</span>
            {/* 문서 요약을 앞세운다. 찾아낸 쪽은 대개 도입부라 그 대목만으로는 내용을 알기 어렵다 */}
            <span className="ctx-snippet">{it.summary?.trim() || plainSnippet(it.text)}</span>
            <span className="ctx-meta">
              <span>
                {[...pages].sort((a, b) => a - b).join("·")}쪽 · 열어 보기
              </span>
              {it.synthetic && <span className="ctx-synthetic">시연용 가상 문서</span>}
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
