#!/usr/bin/env bash
# Ingest an open-source dataset and rebuild the index.
#   ./scripts/ingest.sh --name wikipedia-simple --max-docs 500
#   ./scripts/ingest.sh --hf-path cnn_dailymail --hf-config 3.0.0 --text-field article
#   ./scripts/ingest.sh list      # show available presets
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
activate_venv
cd "$APP_DIR"

if [[ "${1:-}" == "list" ]]; then
  exec python ingest_cli.py list-datasets
fi
exec python ingest_cli.py dataset "$@"
