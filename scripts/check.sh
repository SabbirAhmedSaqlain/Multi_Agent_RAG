#!/usr/bin/env bash
# Full health check: python env, LLM provider reachability, index state.
# Also probes local model servers so you get actionable hints.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
activate_venv

info "Provider configured: ${LLM_PROVIDER:-anthropic}"

probe() { curl -fsS --max-time 3 "$1" >/dev/null 2>&1; }

if probe "http://localhost:11434/v1/models"; then
  info "Ollama detected on :11434 ($(curl -fsS http://localhost:11434/v1/models | python -c 'import sys,json; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"][:5]) or "no models pulled")' 2>/dev/null))"
else
  warn "Ollama not running on :11434 (fine unless LLM_PROVIDER=ollama). Start with: ollama serve"
fi

if probe "http://localhost:1234/v1/models"; then
  info "LM Studio detected on :1234"
else
  warn "LM Studio server not running on :1234 (fine unless LLM_PROVIDER=lmstudio)."
fi

cd "$APP_DIR"
exec python ingest_cli.py check
