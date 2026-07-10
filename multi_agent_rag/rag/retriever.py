from __future__ import annotations
"""
Vector retrieval with three backends:
  numpy  — brute-force cosine (no extra deps, great for prototyping)
  faiss  — IndexFlatIP (exact cosine via normalised inner product, fast in RAM)
  chroma — ChromaDB persistent store (survives restarts, HNSW ANN index)

Set the backend via VECTOR_BACKEND env var or pass backend= to __init__.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from .document_store import Chunk, DocumentStore
from config import EMBEDDING_MODEL, TOP_K_RETRIEVE, TOP_K_FINAL, SCORE_THRESHOLD, VECTOR_BACKEND, CHROMA_PERSIST_DIR, CHROMA_COLLECTION
from utils import get_logger

log = get_logger("VectorRetriever")


class VectorRetriever:
    def __init__(self, store: DocumentStore, backend: str = VECTOR_BACKEND):
        self.store = store
        self.backend = backend
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._chunks: list[Chunk] = []
        self._np_matrix: np.ndarray | None = None
        self._faiss_index = None
        self._chroma_col = None
        self._ready = False
        log.info("VectorRetriever initialised (backend=%s, model=%s)", backend, EMBEDDING_MODEL)

    # ── Public API ─────────────────────────────────────────────────────────

    def build_index(self, embeddings: np.ndarray | None = None):
        """Build the vector index. If `embeddings` is provided (one row per
        chunk, e.g. from IndexManager's embedding cache) the expensive
        encode step is skipped entirely."""
        chunks = self.store.all_chunks()
        if not chunks:
            log.warning("No chunks to index — did you ingest documents?")
            return
        if embeddings is not None and len(embeddings) == len(chunks):
            embs = np.asarray(embeddings)
            log.info("Using %d precomputed embeddings (cache hit)", len(chunks))
        else:
            log.info("Embedding %d chunks...", len(chunks))
            texts = [c.content for c in chunks]
            embs = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=64)

        if self.backend == "numpy":
            self._build_numpy(chunks, embs)
        elif self.backend == "faiss":
            self._build_faiss(chunks, embs)
        elif self.backend == "chroma":
            self._build_chroma(chunks, embs)
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Choose: numpy | faiss | chroma")

        self._chunks = chunks
        self._ready = True
        log.info("Index ready — %d vectors in %s backend", len(chunks), self.backend.upper())

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        if not self._ready:
            log.warning("Index not built yet. Call build_index() first.")
            return []
        q_emb = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)

        if self.backend == "numpy":
            results = self._search_numpy(q_emb, top_k)
        elif self.backend == "faiss":
            results = self._search_faiss(q_emb, top_k)
        elif self.backend == "chroma":
            results = self._search_chroma(query, top_k)
        else:
            results = []

        # Apply score threshold
        results = [r for r in results if r["score"] >= SCORE_THRESHOLD]
        return results[:TOP_K_FINAL]

    # ── NumPy backend ──────────────────────────────────────────────────────

    def _build_numpy(self, chunks: list[Chunk], embs: np.ndarray):
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        self._np_matrix = embs / np.where(norms == 0, 1, norms)
        log.debug("[numpy] Stored %d normalised vectors (dim=%d)", len(chunks), embs.shape[1])

    def _search_numpy(self, q_emb: np.ndarray, top_k: int) -> list[dict]:
        norm = np.linalg.norm(q_emb)
        q_norm = (q_emb / (norm if norm else 1.0)).astype("float32")
        scores = (self._np_matrix @ q_norm.T).squeeze()
        idx = np.argsort(scores)[::-1][:top_k]
        return [self._fmt(self._chunks[i], float(scores[i])) for i in idx]

    # ── FAISS backend ──────────────────────────────────────────────────────

    def _build_faiss(self, chunks: list[Chunk], embs: np.ndarray):
        import faiss
        dim = embs.shape[1]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        normed = (embs / np.where(norms == 0, 1, norms)).astype("float32")
        index = faiss.IndexFlatIP(dim)   # exact cosine via inner product on unit vectors
        index.add(normed)
        self._faiss_index = index
        log.info("[FAISS] IndexFlatIP — %d vectors, dim=%d", index.ntotal, dim)

    def _search_faiss(self, q_emb: np.ndarray, top_k: int) -> list[dict]:
        norm = np.linalg.norm(q_emb)
        q_norm = (q_emb / (norm if norm else 1.0)).astype("float32")
        scores, indices = self._faiss_index.search(q_norm, top_k)
        return [
            self._fmt(self._chunks[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    # ── ChromaDB backend ───────────────────────────────────────────────────

    def _build_chroma(self, chunks: list[Chunk], embs: np.ndarray):
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        col = client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
        batch = 100
        for i in range(0, len(chunks), batch):
            sl = chunks[i:i+batch]
            col.add(
                ids=[c.id for c in sl],
                embeddings=embs[i:i+batch].tolist(),
                documents=[c.content for c in sl],
                metadatas=[{**c.metadata, "doc_id": c.doc_id} for c in sl],
            )
        self._chroma_col = col
        log.info("[ChromaDB] Persisted %d vectors → %s", len(chunks), CHROMA_PERSIST_DIR)

    def _search_chroma(self, query: str, top_k: int) -> list[dict]:
        q_emb = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True).tolist()
        res = self._chroma_col.query(query_embeddings=q_emb, n_results=top_k)
        return [
            {
                "chunk_id": meta.get("chunk_idx", ""),
                "doc_id": meta.get("doc_id", ""),
                "content": doc,
                "score": 1.0 - dist,
                "metadata": meta,
            }
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    # ── Helper ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(chunk: Chunk, score: float) -> dict:
        return {
            "chunk_id": chunk.id,
            "doc_id": chunk.doc_id,
            "content": chunk.content,
            "score": round(score, 4),
            "metadata": chunk.metadata,
        }
