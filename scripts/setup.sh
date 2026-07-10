#!/usr/bin/env bash
# One-time setup: venv + dependencies + config template + provider check.
# Idempotent — safe to re-run any time; already-satisfied steps are skipped.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

info "Setting up Multi-Agent RAG in $REPO_ROOT"

PY="$(find_python)"
info "Using $($PY --version) at $(command -v "$PY")"

# 1. Virtualenv
if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating virtualenv at $VENV_DIR ..."
  "$PY" -m venv "$VENV_DIR"
else
  info "Virtualenv already exists — skipping"
fi
activate_venv

# 2. Dependencies (pip cache makes re-runs fast)
info "Installing dependencies (first run downloads torch — may take a few minutes)..."
pip install --upgrade pip -q
pip install -r "$APP_DIR/requirements.txt" -q || die "Dependency install failed. Check network / try again."
info "Dependencies installed"

# 3. Config template
if [[ ! -f "$REPO_ROOT/.env" ]]; then
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
  warn "Created .env from template — edit it to pick your LLM provider."
else
  info ".env already exists — leaving it untouched"
fi

# 4. Pre-download the embedding model so the first query isn't slow
info "Pre-downloading embedding model (cached under ~/.cache) ..."
python - <<'EOF' || warn "Embedding model pre-download failed (no network?) — it will download on first run instead."
from sentence_transformers import SentenceTransformer
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), ""))
SentenceTransformer(os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
print("Embedding model ready.")
EOF

# 5. Provider status (non-fatal — user may not have configured it yet)
info "Checking LLM provider..."
(cd "$APP_DIR" && python main.py --check) || warn "Provider not ready yet — edit .env (see README 'Choosing an LLM provider')."

echo
info "Setup complete. Next steps:"
echo "    ./scripts/ingest.sh --name wikipedia-simple --max-docs 300   # optional: open-source data"
echo "    ./scripts/run.sh                                             # interactive CLI"
echo "    ./scripts/serve.sh                                           # REST API on :8000"
