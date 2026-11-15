import type { ReactNode } from "react";

/** 답변에 섞여 오는 간단한 마크다운만 그린다: 굵게, 목록, 문단. HTML은 넣지 않는다. */
function inline(text: string): ReactNode[] {
  return text
    .split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
    .filter(Boolean)
    .map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
      if (part.startsWith("`") && part.endsWith("`")) return <span key={i}>{part.slice(1, -1)}</span>;
      return <span key={i}>{part}</span>;
    });
}

export function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let list: string[] = [];
  let para: string[] = [];
  const flushList = () => {
    if (list.length) blocks.push(<ul key={`l${blocks.length}`}>{list.map((t, i) => <li key={i}>{inline(t)}</li>)}</ul>);
    list = [];
  };
  const flushPara = () => {
    if (para.length)
      blocks.push(
        <p key={`p${blocks.length}`}>
          {para.map((t, i) => (
            <span key={i}>
              {i > 0 && <br />}
              {inline(t)}
            </span>
          ))}
        </p>,
      );
    para = [];
  };
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const item = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    if (!line.trim()) {
      flushList();
      flushPara();
    } else if (item) {
      flushPara();
      list.push(item[1]);
    } else if (heading) {
      flushList();
      flushPara();
      blocks.push(
        <p key={`h${blocks.length}`}>
          <strong>{heading[1]}</strong>
        </p>,
      );
    } else {
      flushList();
      para.push(line);
    }
  }
  flushList();
  flushPara();
  return <div className="md">{blocks}</div>;
}
