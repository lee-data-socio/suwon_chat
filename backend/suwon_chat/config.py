"""경로와 설정값. 모델 이름 같은 값은 코드에 박지 않고 환경 변수로 받는다."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """`.env`의 KEY=VALUE를 환경 변수로 올린다. 이미 설정된 값은 건드리지 않는다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


_load_dotenv(ROOT / ".env")

SOURCE_DIR = Path(os.environ.get("SUWON_SOURCE_DIR", ROOT / "수원서베이_학습결과_정리본"))
DB_PATH = Path(os.environ.get("SUWON_DB_PATH", ROOT / "data" / "suwon.db"))
FRONTEND_DIST = ROOT / "frontend" / "dist"

CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-5-5")
CHAT_EFFORT = os.environ.get("CHAT_EFFORT", "low")
CHAT_MAX_TOKENS = int(os.environ.get("CHAT_MAX_TOKENS", "8000"))

PREDICT_YEAR = 2025

# 속성 이름 (자료의 영문 키 → 화면 표기)
ATTRIBUTE_LABELS = {
    "SEX": "성별",
    "AGE": "연령",
    "HOUSING": "주택유형",
    "TENURE": "점유형태",
    "EMPLOYMENT": "고용형태",
    "COMMUTE": "통근·통학",
    "RESIDENCE_YEARS": "거주기간",
    "RELIGION": "종교",
}

# 집단·문항을 빼고 학습한 어댑터를 견줄 기준: 같은 것을 보고 학습한 어댑터
REFERENCE_ADAPTER = "exp1"

# 어댑터 세 개. 학습에서 뺀 집단·문항은 정리본 build_results_v2.py의 EXP2_GROUPS, EXP3_QKEYS와 같다.
ADAPTERS = {
    "exp1": {
        "label": "전체 학습 어댑터",
        "short": "2023·2024년 전체로 학습",
        "description": "2023·2024년 응답으로 학습하고 2025년을 예측한다 (시험 1).",
        "pred_file": "03_예측/시험1_미래예측/pred_bllossom-8b-b128-lr5e-5_suwon-v2-full_1003857.json",
        "held_out_groups": [],
        "held_out_questions": [],
    },
    "exp2": {
        "label": "집단 제외 어댑터",
        "short": "7개 집단을 빼고 학습",
        "description": "속성마다 한 집단씩, 모두 7개 집단의 응답을 한 번도 보여 주지 않고 학습했다 (시험 2).",
        "pred_file": "03_예측/시험2_미관측집단/pred_bllossom-8b-b128-lr5e-5_suwon-v2-exp2_1006787.json",
        "held_out_groups": [
            ["AGE", "50대"], ["HOUSING", "아파트"], ["TENURE", "자가"],
            ["EMPLOYMENT", "상용직 임금근로자"], ["COMMUTE", "통근·통학을 한다"],
            ["RESIDENCE_YEARS", "20년 이상"], ["RELIGION", "없음"],
        ],
        "held_out_questions": [],
    },
    "exp3": {
        "label": "문항 제외 어댑터",
        "short": "4개 문항을 빼고 학습",
        "description": "문항 4개를 모든 연도에서 빼고 학습했다 (시험 3).",
        "pred_file": "03_예측/시험3_새질문/pred_bllossom-8b-b128-lr5e-5_suwon-v2-exp3_1006789.json",
        "held_out_groups": [],
        "held_out_questions": ["q51_1_W2025", "q92_1_W2025", "q120_1_W2025", "q128_2_W2025"],
    },
}
