#!/usr/bin/env bash
# Start the production REST API (uvicorn).
#   ./scripts/serve.sh                 # foreground on API_PORT (default 8000)
#   WORKERS=4 ./scripts/serve.sh       # multiple workers (each builds its own index)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
activate_venv

PORT="${API_PORT:-8000}"
HOST="${API_HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"

# Fail fast if the port is taken
if command -v lsof >/dev/null 2>&1 && lsof -i ":$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  die "Port $PORT is already in use. Stop the other process or set API_PORT."
fi

info "Starting API on http://$HOST:$PORT (workers=$WORKERS, provider=${LLM_PROVIDER:-anthropic})"
info "Docs at http://localhost:$PORT/docs"
cd "$APP_DIR"
exec python -m uvicorn api:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
