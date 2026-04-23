#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

APP_HOST="127.0.0.1"
APP_PORT="5050"
LOG_FILE="logs/server.log"
APP_URL="http://$APP_HOST:$APP_PORT"

mkdir -p logs

# Check if port is already in use
if lsof -nP -iTCP:"$APP_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Error: Port $APP_PORT is already in use."
  echo "Please stop the existing process and try again:"
  echo "  pkill -f 'python app.py'"
  exit 1
fi

echo "[1/5] Checking Python..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 was not found."
  echo "Install Python 3 first, then run this script again."
  exit 1
fi

echo "[2/5] Preparing virtual environment..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "[3/5] Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[4/5] Starting Flask server..."
APP_HOST="$APP_HOST" APP_PORT="$APP_PORT" python app.py > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[5/5] Waiting for server health check..."
for _ in {1..45}; do
  if curl -fsS "$APP_URL/health" >/dev/null 2>&1; then
    open "$APP_URL"
    echo ""
    echo "Server is ready at $APP_URL"
    echo "Log file: $LOG_FILE"
    echo "Press Ctrl+C to stop."
    wait "$SERVER_PID"
    exit 0
  fi
  sleep 1
done

echo "Server did not become ready in time."
echo "Check $LOG_FILE for details."
exit 1