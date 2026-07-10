#!/usr/bin/env bash
# Regular data update job — designed for cron / launchd / CI schedules.
#   Re-pulls every previously ingested dataset (new records only, idempotent)
#   then incrementally re-indexes (only new chunks get embedded).
#
# Cron example (hourly, with log):
#   0 * * * *  /path/to/repo/scripts/update_data.sh >> /path/to/repo/multi_agent_rag/logs/update.log 2>&1
#
# A lock file prevents overlapping runs if one update takes longer than the interval.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
activate_venv

LOCK_FILE="$REPO_ROOT/.update.lock"
if [[ -f "$LOCK_FILE" ]]; then
  pid="$(cat "$LOCK_FILE" 2>/dev/null || echo "")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    warn "Update already running (pid $pid) — skipping this run."
    exit 0
  fi
  warn "Stale lock found — removing."
  rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

info "[$(date '+%Y-%m-%d %H:%M:%S')] Data update starting..."
cd "$APP_DIR"
python ingest_cli.py refresh "$@"
rc=$?

if [[ -n "${RAG_API_URL:-}" ]]; then
  # If the API server is running, tell it to hot-swap its index (no restart)
  info "Notifying API at $RAG_API_URL to refresh its in-memory index..."
  curl -fsS -X POST "$RAG_API_URL/refresh" \
       -H 'Content-Type: application/json' \
       -d '{"pull_datasets": false}' >/dev/null \
    && info "API refresh scheduled" \
    || warn "Could not reach API — it will pick up changes on next restart."
fi

info "[$(date '+%Y-%m-%d %H:%M:%S')] Data update finished (rc=$rc)"
exit $rc
