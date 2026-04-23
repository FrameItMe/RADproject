#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  osascript -e 'display dialog "Python 3 was not found. Install Python 3 first." buttons {"OK"} default button "OK"'
  exit 1
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt >/dev/null

mkdir -p logs
python app.py > logs/server.log 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in {1..30}; do
  if curl -fsS http://127.0.0.1:5050/health >/dev/null 2>&1; then
    open "http://127.0.0.1:5050"
    echo "Server ready."
    wait "$SERVER_PID"
    exit 0
  fi
  sleep 1
done

osascript -e 'display dialog "The app started, but the server did not become ready in time. Check logs/server.log." buttons {"OK"} default button "OK"'
exit 1