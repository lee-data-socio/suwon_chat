#!/usr/bin/env bash
# 앱 서버를 띄운다. 색인(예측 DB, 참고 문서)이 없으면 먼저 만든다.
#   scripts/serve.sh [포트]
set -euo pipefail
cd "$(dirname "$0")/../backend"
PORT="${1:-8000}"
PYTHON="${PYTHON:-python}"
if [ ! -f ../data/suwon.db ]; then
  "$PYTHON" -m suwon_chat.build_index
fi
# 참고 문서 색인이 없으면 답변의 해석과 문서 카드가 조용히 빠진다. 없으면 만든다
if [ ! -f ../data/reports_index.json ]; then
  "$PYTHON" -m suwon_chat.reports
fi
# 문항 검색 색인이 없으면 담당자식 질문을 단어 검색으로만 찾는다. 없으면 만든다
if [ ! -f ../data/concepts_index.json ]; then
  "$PYTHON" -m suwon_chat.concepts
fi
exec "$PYTHON" -m uvicorn suwon_chat.api:app --host 127.0.0.1 --port "$PORT"
