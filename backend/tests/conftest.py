import csv
import os
import sys
from pathlib import Path

os.environ["SUWON_WARM_EMBEDDER"] = "0"      # 테스트에서는 임베딩 모델을 올리지 않는다

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


# ---- 참고 문서: 테스트용 문서와 가짜 임베딩 (모델을 올리지 않는다) -------------------
import numpy as np  # noqa: E402

from suwon_chat.reports import ReportIndex, build_index  # noqa: E402

DOCS = {
    "R01_bus.md": """---
id: R01
title: "[시연용 가상 문서] 대중교통 인프라와 버스 만족도"
summary: "버스 인프라에 대한 인식을 다룬다."
topics: ["버스", "교통"]
---
<!-- page 1 -->
# 대중교통 인프라와 버스 만족도

수원시 교통 수단 가운데 버스(시내, 마을버스 등) 인프라가 얼마나 잘 구축되어 있다고 생각하는지 물은 문항을 본다.

<!-- page 2 -->
## 집단별 차이

버스 인프라 평가는 점유형태에 따라 조금씩 달랐다.
""",
    "R02_class.md": """---
id: R02
title: "[시연용 가상 문서] 주관적 계층 인식"
summary: "본인의 경제·사회적 위치에 대한 인식을 다룬다."
topics: ["계층"]
---
<!-- page 1 -->
# 주관적 계층 인식

한국사회의 최하층을 0으로 하고 최상층을 10으로 할 때 본인의 경제·사회적 위치가 어디에 해당된다고 생각하는지 물었다.
""",
    "R03_real.md": """---
id: R03
title: "반려동물 양육 실태"
summary: "반려동물이 있는 가구를 다룬다."
topics: ["반려동물"]
---
<!-- page 1 -->
# 반려동물 양육 실태

귀댁에는 반려동물이 있습니까라는 문항의 응답을 정리했다.
""",
}


def fake_embed(texts, kind):
    """글자 두 개씩 묶어 센 벡터. 같은 낱말이 많을수록 가깝다."""
    out = np.zeros((len(texts), 512), np.float32)
    for row, text in zip(out, texts):
        for a, b in zip(text, text[1:]):
            row[(ord(a) * 31 + ord(b)) % 512] += 1
        row /= max(np.linalg.norm(row), 1e-9)
    return out


@pytest.fixture(scope="session")
def report_index(tmp_path_factory, service) -> ReportIndex:
    folder = tmp_path_factory.mktemp("reports")
    for name, text in DOCS.items():
        (folder / name).write_text(text, encoding="utf-8")
    out = tmp_path_factory.mktemp("report_index")
    questions = {q.qkey: q.text for q in service.store.target_questions}
    build_index(folder, questions, fake_embed, out / "index.json", out / "vectors.npy")
    # 가짜 임베딩은 점수 분포가 실제 모델과 달라 문턱을 끈다. 문턱은 test_reports에서 따로 본다.
    return ReportIndex(out / "index.json", out / "vectors.npy", fake_embed, min_score=0.0, margin=1.0)
