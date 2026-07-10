"""
Production REST API for the Multi-Agent RAG system (FastAPI).

Run:  uvicorn api:app --host 0.0.0.0 --port 8000        (from multi_agent_rag/)
 or:  ./scripts/serve.sh                                  (from repo root)

Endpoints
  GET  /health            liveness + provider + index status
  GET  /stats             knowledge-base statistics
  GET  /datasets          available open-source dataset presets
  POST /query             run the full multi-agent pipeline
  POST /ingest/text       add a document inline
  POST /ingest/dataset    ingest an open-source dataset (background)
  POST /refresh           re-pull datasets + incremental re-index (background)

Concurrency model: queries hold a read snapshot of the current retriever;
re-indexing builds a NEW IndexManager and atomically swaps it in, so queries
never see a half-built index.
"""
import sys
import os
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

import config
from llm import check_provider, LLMError
from rag import IndexManager
from rag.dataset_loader import list_presets, load_preset, refresh_all
from agents import run_pipeline
from utils import get_logger

log = get_logger("API")

_manager: IndexManager | None = None
_swap_lock = threading.Lock()        # protects _manager swaps
_reindex_lock = threading.Lock()     # only one background re-index at a time


def _get_manager() -> IndexManager:
    if _manager is None or not _manager.is_ready():
        raise HTTPException(status_code=503, detail="Index not ready yet — try again shortly.")
    return _manager


def _rebuild_and_swap(force: bool = False) -> dict:
    """Build a fresh index off to the side, then swap atomically."""
    global _manager
    if not _reindex_lock.acquire(blocking=False):
        return {"status": "skipped", "reason": "another re-index is already running"}
    try:
        new_mgr = IndexManager()
        report = new_mgr.sync(force=force)
        with _swap_lock:
            _manager = new_mgr
        return {"status": "ok", **report}
    finally:
        _reindex_lock.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager
    log.info("Starting up: building index (provider=%s, backend=%s)",
             config.LLM_PROVIDER, config.VECTOR_BACKEND)
    mgr = IndexManager()
    mgr.sync()
    _manager = mgr
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Multi-Agent RAG API",
    description="Production multi-agent RAG: QueryAnalyzer → Retriever → Analyzer → Synthesizer → Critic",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=4000)


class IngestTextRequest(BaseModel):
    content: str = Field(..., min_length=20)
    source: str = Field("api-upload", max_length=200)


class IngestDatasetRequest(BaseModel):
    name: str = Field(..., description="Preset name — see GET /datasets")
    max_docs: int = Field(config.DATASET_MAX_DOCS, ge=1, le=100_000)


class RefreshRequest(BaseModel):
    pull_datasets: bool = Field(True, description="Re-pull previously ingested datasets first")
    force: bool = Field(False, description="Ignore the embedding cache (full re-embed)")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    provider = check_provider()
    index_ready = _manager is not None and _manager.is_ready()
    status = "ok" if provider["ok"] and index_ready else "degraded"
    return {
        "status": status,
        "provider": provider,
        "index": _manager.stats() if index_ready else {"synced": False},
    }


@app.get("/stats")
def stats():
    return _get_manager().stats()


@app.get("/datasets")
def datasets():
    return list_presets()


@app.post("/query")
def query(req: QueryRequest):
    mgr = _get_manager()
    try:
        result = run_pipeline(req.query, mgr.retriever)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return result


@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest, background: BackgroundTasks):
    # Persist as a corpus file so it survives restarts and syncs like any doc
    import hashlib
    h = hashlib.sha256(req.content.encode()).hexdigest()[:16]
    out_dir = config.CORPUS_DIR / "api_uploads"
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{h}.txt"
    created = not fpath.exists()
    if created:
        fpath.write_text(f"{req.source}\n\n{req.content}", encoding="utf-8")
        background.add_task(_rebuild_and_swap)
    return {"status": "ok", "created": created, "file": str(fpath),
            "note": "index refresh scheduled" if created else "duplicate — already indexed"}


@app.post("/ingest/dataset")
def ingest_dataset(req: IngestDatasetRequest, background: BackgroundTasks):
    if req.name not in list_presets():
        raise HTTPException(status_code=400,
                            detail=f"Unknown preset '{req.name}'. See GET /datasets.")

    def job():
        try:
            stats = load_preset(req.name, max_docs=req.max_docs)
            log.info("Background dataset ingestion done: %s", stats)
            _rebuild_and_swap()
        except Exception as e:  # noqa: BLE001 — background boundary
            log.error("Background dataset ingestion failed: %s", e)

    background.add_task(job)
    return {"status": "scheduled", "dataset": req.name, "max_docs": req.max_docs,
            "note": "ingestion + re-index running in background; watch GET /stats"}


@app.post("/refresh")
def refresh(req: RefreshRequest, background: BackgroundTasks):
    def job():
        try:
            if req.pull_datasets:
                refresh_all()
            result = _rebuild_and_swap(force=req.force)
            log.info("Background refresh done: %s", result)
        except Exception as e:  # noqa: BLE001 — background boundary
            log.error("Background refresh failed: %s", e)

    background.add_task(job)
    return {"status": "scheduled",
            "note": "refresh running in background; watch GET /stats"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
