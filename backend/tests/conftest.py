import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suwon_chat import config  # noqa: E402
from suwon_chat.build_index import build  # noqa: E402
from suwon_chat.predict import LookupBackend  # noqa: E402
from suwon_chat.service import Service  # noqa: E402
from suwon_chat.store import Store  # noqa: E402

RESULTS = config.SOURCE_DIR / "01_결과표"


@pytest.fixture(scope="session")
def service(tmp_path_factory) -> Service:
    """정리본에서 새로 색인한 DB로 서비스를 만든다. 색인의 자체 점검도 이때 함께 돈다."""
    db = tmp_path_factory.mktemp("db") / "suwon.db"
    build(config.SOURCE_DIR, db)
    store = Store(db)
    return Service(store, LookupBackend(store))


def result_rows(experiment: str, run_contains: str) -> list[dict]:
    """정리본 결과표의 칸 단위 값. 서비스가 계산한 값과 견줄 기댓값으로 쓴다."""
    with open(RESULTS / experiment / "칸단위.csv", encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if run_contains in r["run"]]
