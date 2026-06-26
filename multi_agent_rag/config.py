import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── API ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-opus-4-8"

# ── Agent settings ─────────────────────────────────────────────────────────
MAX_TOKENS = 8192
MAX_REVISION_CYCLES = 2           # critic can trigger at most N re-syntheses

# ── RAG settings ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600                  # characters per chunk
CHUNK_OVERLAP = 80                # overlap between consecutive chunks
TOP_K_RETRIEVE = 8                # retrieved before reranking
TOP_K_FINAL = 5                   # kept after reranking
SCORE_THRESHOLD = 0.25            # minimum cosine similarity to keep a chunk

# ── Vector store backend ───────────────────────────────────────────────────
# Options: "faiss" | "chroma" | "numpy"
VECTOR_BACKEND: str = os.environ.get("VECTOR_BACKEND", "faiss")
CHROMA_COLLECTION = "production_rag"

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
