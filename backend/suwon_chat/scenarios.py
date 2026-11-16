"""추천 질문. 목록은 옆의 `scenarios.json`에 있고, 그 파일만 고치면 버튼이 바뀐다.

버튼을 누르면 패널이 `panel` 상태로 바뀌고 `prompt`가 채팅으로 간다. 패널 상태가 파일에 박혀 있으므로
대화 LLM이나 인터넷이 없어도 패널은 돈다. `compare`는 함께 보일 비교 대상이다:
actual(2025년 실측) · last_year(2024년 실측) · reference(전체 학습 어댑터).
"""
import json
from pathlib import Path

SCENARIOS = json.loads((Path(__file__).with_name("scenarios.json")).read_text(encoding="utf-8"))
