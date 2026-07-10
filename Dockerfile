# ── Production image for the Multi-Agent RAG API ───────────────────────────
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# curl for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached layer — rebuilds only when requirements change)
COPY multi_agent_rag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY multi_agent_rag/ ./multi_agent_rag/

# Pre-bake the embedding model so cold starts don't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Non-root user; writable dirs for corpus / index state / logs / chroma
RUN useradd -m raguser \
    && mkdir -p /app/multi_agent_rag/corpus /app/multi_agent_rag/index_state \
                /app/multi_agent_rag/logs /app/multi_agent_rag/chroma_db \
    && chown -R raguser:raguser /app
USER raguser

WORKDIR /app/multi_agent_rag
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
