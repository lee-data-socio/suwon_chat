"""예측 백엔드. 화면과 대화 LLM은 `predict(문항, 속성, 어댑터)` 하나만 안다.

조회 백엔드는 정리본에 저장된 예측을 돌려준다. 실시간 백엔드는 같은 모양으로
모델을 직접 돌린 값을 돌려주게 된다. 어느 쪽인지는 `source`로 화면에 표시한다.
"""
from __future__ import annotations

from typing import Protocol

from .store import Store


class PredictionBackend(Protocol):
    source: str  # "precomputed" | "live"

    def predict(self, qkey: str, attribute: str, adapter: str) -> dict[str, dict[str, list[float]]]:
        """집단 → {"lora": 어댑터 예측, "base": 학습 전 모델 예측}. 무응답 칸이 붙은 원래 길이 그대로."""
        ...


class LookupBackend:
    source = "precomputed"

    def __init__(self, store: Store):
        self.store = store

    def predict(self, qkey: str, attribute: str, adapter: str) -> dict[str, dict[str, list[float]]]:
        out = {}
        for group in self.store.groups[attribute]:
            row = self.store.predictions.get((adapter, qkey, attribute, group))
            if row is None:
                raise KeyError(f"저장된 예측이 없다: {adapter} / {qkey} / {attribute} / {group}")
            out[group] = {"lora": row["lora"], "base": row["base"]}
        return out
