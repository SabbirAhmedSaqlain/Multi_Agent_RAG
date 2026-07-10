#!/usr/bin/env bash
# Shared helpers for all scripts — source this, don't run it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$REPO_ROOT/multi_agent_rag"
VENV_DIR="$REPO_ROOT/.venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()   { error "$*"; exit 1; }

trap 'error "Script failed at line $LINENO (exit code $?)"' ERR

# Load .env if present so scripts see the same config as the app
load_env() {
  if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"; set +a
    info "Loaded .env"
  fi
}

find_python() {
  for cand in python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      ver="$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      major="${ver%%.*}"; minor="${ver##*.}"
      if [[ "$major" -eq 3 && "$minor" -ge 10 ]]; then
        echo "$cand"; return 0
      fi
    fi
  done
  die "Python 3.10+ not found. Install it from https://www.python.org or via 'brew install python@3.12'."
}

activate_venv() {
  [[ -d "$VENV_DIR" ]] || die "Virtualenv missing. Run: ./scripts/setup.sh"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
}
