import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { fetchReport } from "../api";
import type { ContextItem, Report } from "../types";
import { cleanTitle } from "../lib/format";
import { Markdown } from "../lib/markdown";

/** 참고 문서의 전문. 인용된 쪽을 모두 표시해 두고, 그 가운데 가장 앞쪽으로 내려가 연다. */
export function ReportModal({ item, pages: citedPages, onClose }: { item: ContextItem; pages: number[]; onClose: () => void }) {
  const isCited = (page: number) => citedPages.includes(page);
  const firstCited = Math.min(...citedPages);
  const [report, setReport] = useState<Report | null>(null);
  const [failed, setFailed] = useState(false);
  const cited = useRef<HTMLElement | null>(null);

  useEffect(() => {
    let alive = true;
    fetchReport(item.doc_id)
      .then((r) => alive && setReport(r))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [item.doc_id]);

  useEffect(() => {
    cited.current?.scrollIntoView({ block: "start" });
  }, [report]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // 전문을 못 가져와도 인용된 대목은 보인다
  const pages = report?.pages ?? [{ page: item.page, text: item.text }];

  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="report" role="dialog" aria-modal="true" aria-label="참고 문서">
        <header className="report-head">
          <div>
            <div className="report-kind">
              참고 문서{item.synthetic && <span className="ctx-synthetic">시연용 가상 문서</span>}
            </div>
            <h2>{cleanTitle(report?.title ?? item.title)}</h2>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="닫기">
            <X size={18} />
          </button>
        </header>
        <div className="report-body">
          {pages.map((pg) => (
            <section
              key={pg.page}
              className={`report-page${isCited(pg.page) ? " is-cited" : ""}`}
              ref={pg.page === firstCited ? (el) => void (cited.current = el) : undefined}
            >
              <div className="report-pageno">
                {pg.page}쪽{isCited(pg.page) && <span>인용된 쪽</span>}
              </div>
              <Markdown text={pg.text} />
            </section>
          ))}
          {failed && <p className="report-fail">문서 전문을 불러오지 못해 인용된 쪽만 보입니다.</p>}
        </div>
        <footer className="report-foot">
          {item.synthetic ? "시연을 위해 만든 가상 문서입니다. " : ""}이 문서는 예측에 사용되지 않았습니다.
        </footer>
      </div>
    </div>
  );
}
