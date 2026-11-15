import { createContext, useCallback, useContext, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

interface TipState {
  x: number;
  y: number;
  content: ReactNode;
}

interface TipApi {
  show: (e: { clientX: number; clientY: number } | Element, content: ReactNode) => void;
  hide: () => void;
}

const TipContext = createContext<TipApi>({ show: () => {}, hide: () => {} });

export const useTip = () => useContext(TipContext);

/** 화면 전체에 하나만 있는 말풍선. 마우스를 올리거나 키보드로 초점을 주면 같은 내용이 나온다. */
export function TipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const ref = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState({ left: 0, top: 0 });

  const api = useMemo<TipApi>(
    () => ({
      show: (e, content) => {
        if (e instanceof Element) {
          const r = e.getBoundingClientRect();
          setTip({ x: r.left + r.width / 2, y: r.bottom, content });
        } else setTip({ x: e.clientX, y: e.clientY, content });
      },
      hide: () => setTip(null),
    }),
    [],
  );

  useLayoutEffect(() => {
    if (!tip || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    let left = tip.x + 14;
    let top = tip.y + 16;
    if (left + r.width > window.innerWidth - 8) left = tip.x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = tip.y - r.height - 12;
    setPos({ left: Math.max(8, left), top: Math.max(8, top) });
  }, [tip]);

  return (
    <TipContext.Provider value={api}>
      {children}
      {tip && (
        <div ref={ref} className="tip" role="tooltip" style={{ left: pos.left, top: pos.top }}>
          {tip.content}
        </div>
      )}
    </TipContext.Provider>
  );
}

/** 요소에 붙이는 말풍선 손잡이 (마우스와 키보드 초점 둘 다). */
export function useTipHandlers(content: () => ReactNode) {
  const tip = useTip();
  const onPointerMove = useCallback((e: React.PointerEvent) => tip.show(e, content()), [tip, content]);
  const onFocus = useCallback((e: React.FocusEvent) => tip.show(e.currentTarget, content()), [tip, content]);
  return { onPointerMove, onPointerLeave: tip.hide, onFocus, onBlur: tip.hide };
}

/** 짧은 설명을 띄우는 물음표 없는 도움말: 글자에 점선 밑줄 대신 말풍선만 붙인다. */
export function Hint({ text, children, className }: { text: ReactNode; children: ReactNode; className?: string }) {
  const tip = useTip();
  return (
    <span
      className={className}
      tabIndex={0}
      onPointerMove={(e) => tip.show(e, <div className="tip-text">{text}</div>)}
      onPointerLeave={tip.hide}
      onFocus={(e) => tip.show(e.currentTarget, <div className="tip-text">{text}</div>)}
      onBlur={tip.hide}
    >
      {children}
    </span>
  );
}
