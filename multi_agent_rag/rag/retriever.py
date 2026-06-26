import numpy as np
from sentence_transformers import SentenceTransformer
from .document_store import Chunk, DocumentStore
from config import EMBEDDING_MODEL, TOP_K_RESULTS


class VectorRetriever:
    def __init__(self, document_store: DocumentStore):
        self.store = document_store
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._embeddings: np.ndarray | None = None
        self._indexed_chunks: list[Chunk] = []

    def build_index(self):
        chunks = self.store.get_all_chunks()
        if not chunks:
            return
        texts = [c.content for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        self._embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._indexed_chunks = chunks

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        if self._embeddings is None or len(self._indexed_chunks) == 0:
            return []

        q_emb = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        scores = (self._embeddings @ q_emb.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self._indexed_chunks[idx]
            results.append({
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "score": float(scores[idx]),
                "metadata": chunk.metadata,
            })
        return results
