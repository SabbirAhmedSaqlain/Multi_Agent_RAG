"""
Central configuration.

Every setting can be overridden with an environment variable (or a `.env`
file in the repo root / package dir). Defaults are chosen so the system
runs out-of-the-box with the bundled sample data.
"""
import os
from pathlib import Path

# ── .env loading (optional, no hard dependency) ────────────────────────────
BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent


def _load_dotenv() -> None:
    """Load .env from repo root or package dir. Uses python-dotenv when
    available, falls back to a minimal parser. Never raises."""
    candidates = [REPO_ROOT / ".env", BASE_DIR / ".env"]
    try:
        from dotenv import load_dotenv
        for p in candidates:
            if p.exists():
                load_dotenv(p, override=False)
        return
    except ImportError:
        pass
    for p in candidates:
        if not p.exists():
            continue
        try:
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip("'\"")
                os.environ.setdefault(key, val)
        except OSError:
            pass


_load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", BASE_DIR / "corpus"))       # dataset-materialised docs
INDEX_STATE_DIR = Path(os.environ.get("INDEX_STATE_DIR", BASE_DIR / "index_state"))  # manifest + embedding cache
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
LOG_DIR = BASE_DIR / "logs"
for _d in (LOG_DIR, CORPUS_DIR, INDEX_STATE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── LLM provider ───────────────────────────────────────────────────────────
# Options:
#   anthropic — Claude via the Anthropic API (needs ANTHROPIC_API_KEY)
#   ollama    — local open-source models via Ollama  (http://localhost:11434)
#   lmstudio  — local open-source models via LM Studio (http://localhost:1234)
#   openai    — ANY OpenAI-compatible endpoint (vLLM, llama.cpp server,
#               Together, Groq, OpenRouter, ...) via OPENAI_BASE_URL
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()

# Anthropic
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")

# Ollama (OpenAI-compatible endpoint)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# LM Studio (OpenAI-compatible endpoint)
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "")   # empty = first loaded model

# Generic OpenAI-compatible endpoint
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Shared LLM behaviour
MAX_TOKENS = _env_int("MAX_TOKENS", 8192)
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.2)
LLM_TIMEOUT = _env_float("LLM_TIMEOUT", 300.0)          # seconds per call
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 3)
LLM_RETRY_BACKOFF = _env_float("LLM_RETRY_BACKOFF", 2.0)  # base seconds, exponential

# ── Agent settings ─────────────────────────────────────────────────────────
MAX_REVISION_CYCLES = _env_int("MAX_REVISION_CYCLES", 2)  # critic can trigger at most N re-syntheses

# ── RAG settings ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = _env_int("CHUNK_SIZE", 600)                # characters per chunk
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 80)           # overlap between consecutive chunks
TOP_K_RETRIEVE = _env_int("TOP_K_RETRIEVE", 8)          # retrieved before reranking
TOP_K_FINAL = _env_int("TOP_K_FINAL", 5)                # kept after reranking
SCORE_THRESHOLD = _env_float("SCORE_THRESHOLD", 0.25)   # minimum cosine similarity to keep a chunk

# ── Vector store backend ───────────────────────────────────────────────────
# Options: "faiss" | "chroma" | "numpy"
VECTOR_BACKEND: str = os.environ.get("VECTOR_BACKEND", "faiss")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "production_rag")

# ── Open-source dataset ingestion (Hugging Face) ───────────────────────────
DATASET_NAME = os.environ.get("DATASET_NAME", "wikipedia-simple")
DATASET_MAX_DOCS = _env_int("DATASET_MAX_DOCS", 500)
DATASET_MIN_CHARS = _env_int("DATASET_MIN_CHARS", 200)  # skip trivially short records

# ── API server ─────────────────────────────────────────────────────────────
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8000)

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
