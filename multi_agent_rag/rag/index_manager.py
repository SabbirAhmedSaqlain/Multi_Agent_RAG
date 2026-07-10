"""
IndexManager — the single entry point for building and *keeping up to date*
the knowledge base.

Regular data injection / update model
─────────────────────────────────────
Documents live as plain files in two places:
  DATA_DIR    — hand-curated files you drop in (.txt / .md)
  CORPUS_DIR  — files materialised from open-source datasets (dataset_loader)

`sync()` scans both, compares content hashes against the saved manifest
(index_state/manifest.json), and rebuilds the vector index. The expensive
part — embedding — is cached per-chunk in index_state/embeddings.npz keyed
by content hash, so a sync after adding 10 documents to a 10,000-document
corpus embeds only the new chunks. Deleted/changed files drop out naturally
because the store is rebuilt from what is on disk (single source of truth).

Typical cadence:
  - on process start:            sync()          (fast — cache hits)
  - after dataset refresh/cron:  sync()          (embeds only new docs)
  - after changing chunking/embedding settings:  sync(force=True)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from config import DATA_DIR, CORPUS_DIR, INDEX_STATE_DIR, VECTOR_BACKEND, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from utils import get_logger
from .document_store import DocumentStore
from .ingestion import Ingestion
from .retriever import VectorRetriever

log = get_logger("IndexManager")

MANIFEST_PATH = INDEX_STATE_DIR / "manifest.json"
EMB_CACHE_PATH = INDEX_STATE_DIR / "embeddings.npz"


def _file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


class IndexManager:
    def __init__(self, backend: str = VECTOR_BACKEND):
        self.backend = backend
        self.store = DocumentStore()
        self.ingestion = Ingestion(self.store)
        self.retriever = VectorRetriever(self.store, backend=backend)
        self._synced = False

    # ── Public API ──────────────────────────────────────────────────────────

    def sync(self, force: bool = False) -> dict:
        """Scan DATA_DIR + CORPUS_DIR, (re)build the index, embedding only
        chunks not present in the cache. Returns a change report."""
        t0 = time.time()
        files = self._discover_files()
        manifest_old = self._load_manifest()
        manifest_new = {str(p): _file_hash(p) for p in files}

        added = [p for p in manifest_new if p not in manifest_old]
        changed = [p for p in manifest_new
                   if p in manifest_old and manifest_old[p] != manifest_new[p]]
        removed = [p for p in manifest_old if p not in manifest_new]

        log.info("Sync: %d files (%d new, %d changed, %d removed)%s",
                 len(files), len(added), len(changed), len(removed),
                 " [force]" if force else "")

        # Rebuild the in-memory store from disk (single source of truth)
        self.store = DocumentStore()
        self.ingestion = Ingestion(self.store)
        self.retriever = VectorRetriever(self.store, backend=self.backend)
        for d in (DATA_DIR, CORPUS_DIR):
            if Path(d).exists():
                self.ingestion.ingest_directory(str(d), extensions=[".txt", ".md"],
                                                recursive=True)

        chunks = self.store.all_chunks()
        embeddings, cache_hits = self._embed_with_cache(chunks, force=force)
        self.retriever.build_index(embeddings=embeddings)

        self._save_manifest(manifest_new)
        self._synced = True

        report = {
            "files": len(files),
            "added": len(added),
            "changed": len(changed),
            "removed": len(removed),
            "documents": self.store.stats()["documents"],
            "chunks": len(chunks),
            "embedded_new": len(chunks) - cache_hits,
            "embedded_cached": cache_hits,
            "backend": self.backend,
            "seconds": round(time.time() - t0, 2),
        }
        log.info("Sync complete: %s", report)
        return report

    def is_ready(self) -> bool:
        return self._synced

    def stats(self) -> dict:
        s = self.store.stats()
        s.update({
            "backend": self.backend,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "synced": self._synced,
        })
        return s

    # ── Internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _discover_files() -> list[Path]:
        files: list[Path] = []
        for d in (DATA_DIR, CORPUS_DIR):
            root = Path(d)
            if root.exists():
                files.extend(p for p in root.rglob("*")
                             if p.is_file() and p.suffix in (".txt", ".md")
                             and not p.name.startswith("_"))
        return sorted(files)

    @staticmethod
    def _load_manifest() -> dict:
        if MANIFEST_PATH.exists():
            try:
                return json.loads(MANIFEST_PATH.read_text())
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Manifest unreadable (%s) — treating all files as new", e)
        return {}

    @staticmethod
    def _save_manifest(manifest: dict) -> None:
        tmp = MANIFEST_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(MANIFEST_PATH)  # atomic — a crash never corrupts the manifest

    def _embed_with_cache(self, chunks, force: bool = False) -> tuple[np.ndarray | None, int]:
        """Return embeddings for all chunks, computing only cache misses.
        Cache key = sha256(chunk content) so it survives file moves/renames
        and is invalidated automatically when chunking parameters change
        (different chunk text ⇒ different hash)."""
        if not chunks:
            return None, 0

        cache: dict[str, np.ndarray] = {}
        if EMB_CACHE_PATH.exists() and not force:
            try:
                with np.load(EMB_CACHE_PATH) as npz:
                    cache = {k: npz[k] for k in npz.files}
            except Exception as e:  # noqa: BLE001 — corrupt cache = re-embed, not crash
                log.warning("Embedding cache unreadable (%s) — re-embedding all", e)
                cache = {}

        hashes = [_chunk_hash(c.content) for c in chunks]
        miss_idx = [i for i, h in enumerate(hashes) if h not in cache]
        hits = len(chunks) - len(miss_idx)

        if miss_idx:
            log.info("Embedding %d new chunks (%d cached)...", len(miss_idx), hits)
            texts = [chunks[i].content for i in miss_idx]
            new_embs = self.retriever.model.encode(
                texts, show_progress_bar=len(texts) > 200,
                convert_to_numpy=True, batch_size=64,
            )
            for i, emb in zip(miss_idx, new_embs):
                cache[hashes[i]] = emb.astype("float32")

        embeddings = np.stack([cache[h] for h in hashes]).astype("float32")

        # Persist only hashes still in use — the cache never grows unbounded
        live = {h: cache[h] for h in set(hashes)}
        try:
            # savez appends ".npz" unless the name already ends with it
            tmp = EMB_CACHE_PATH.with_name("embeddings_tmp.npz")
            np.savez_compressed(tmp, **live)
            tmp.replace(EMB_CACHE_PATH)
        except OSError as e:
            log.warning("Could not persist embedding cache: %s", e)

        return embeddings, hits
