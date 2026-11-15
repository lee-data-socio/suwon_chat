#!/usr/bin/env bash
# 앱 서버를 띄운다. 색인 DB가 없으면 먼저 만든다.
#   scripts/serve.sh [포트]
set -euo pipefail
cd "$(dirname "$0")/../backend"
PORT="${1:-8000}"
PYTHON="${PYTHON:-python}"
if [ ! -f ../data/suwon.db ]; then
  "$PYTHON" -m suwon_chat.build_index
fi
exec "$PYTHON" -m uvicorn suwon_chat.api:app --host 127.0.0.1 --port "$PORT"
