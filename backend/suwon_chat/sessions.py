"""대화 기록. 서버 메모리에만 둔다. 서버를 끄면 모두 사라진다 (시연용: 파일이나 DB에 남기지 않는다).

화면이 보낸 메시지 목록을 그대로 보관했다가 돌려준다. 서버는 내용을 해석하지 않는다.
"""
from __future__ import annotations

import threading
import time

MAX_SESSIONS = 100        # 넘으면 가장 오래 손대지 않은 대화부터 버린다


class SessionStore:
    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self.max_sessions = max_sessions
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}

    def save(self, sid: str, title: str, messages: list[dict]) -> dict:
        now = time.time()
        with self._lock:
            old = self._items.get(sid)
            item = {"id": sid, "title": title.strip() or "새 대화", "messages": messages,
                    "created_at": old["created_at"] if old else now, "updated_at": now}
            self._items[sid] = item
            if len(self._items) > self.max_sessions:
                oldest = min(self._items.values(), key=lambda s: s["updated_at"])
                del self._items[oldest["id"]]
            return self._summary(item)

    def get(self, sid: str) -> dict | None:
        with self._lock:
            item = self._items.get(sid)
            return dict(item) if item else None

    def delete(self, sid: str) -> bool:
        with self._lock:
            return self._items.pop(sid, None) is not None

    def listing(self) -> list[dict]:
        """최근에 손댄 대화부터"""
        with self._lock:
            items = sorted(self._items.values(), key=lambda s: s["updated_at"], reverse=True)
            return [self._summary(s) for s in items]

    @staticmethod
    def _summary(s: dict) -> dict:
        return {"id": s["id"], "title": s["title"], "updated_at": s["updated_at"],
                "count": sum(1 for m in s["messages"] if m.get("role") == "user")}
