#!/usr/bin/env bash
# Run the interactive CLI (or one-shot with --query "...").
# All arguments are passed through to main.py, e.g.:
#   ./scripts/run.sh --query "How does CRISPR work?"
#   ./scripts/run.sh --dataset ag-news --max-docs 200
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
activate_venv
cd "$APP_DIR"
exec python main.py "$@"
