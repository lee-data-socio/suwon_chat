import { useEffect, useState } from "react";
import { cachedPrediction, fetchPrediction } from "../api";
import type { PanelState, Prediction } from "../types";

/** 문항 × 속성 × 어댑터의 예측을 가져온다. 대화가 보낸 자료는 이미 저장돼 있어 바로 나온다. */
export function usePrediction(state: PanelState | null) {
  const key = state ? `${state.qkey}|${state.attribute}|${state.adapter}` : "";
  const [data, setData] = useState<Prediction | null>(() => (state ? (cachedPrediction(state) ?? null) : null));
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!state) return;
    const hit = cachedPrediction(state);
    if (hit) {
      setData(hit);
      setError(null);
      setLoading(false);
      return;
    }
    let alive = true;
    setLoading(true);
    fetchPrediction(state)
      .then((p) => {
        if (!alive) return;
        setData(p);
        setError(null);
      })
      .catch((e: Error) => {
        if (!alive) return;
        setData(null); // 이전 장면의 자료를 남겨 두지 않는다
        setError(`예측을 불러오지 못했습니다: ${e.message}`);
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // key가 같으면 같은 장면이다
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return { data, error, loading };
}
