from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils import get_logger

log = get_logger("DocumentStore")


@dataclass
class Document:
    id: str
    content: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list] = None


class DocumentStore:
    def __init__(self):
        self._documents: dict[str, Document] = {}
        self._chunks: list[Chunk] = []

    # ── Ingestion ──────────────────────────────────────────────────────────

    def add_text(self, text: str, source: str = "", metadata: dict | None = None) -> str:
        doc_id = self._make_id(text, source)
        if doc_id in self._documents:
            log.debug("Document already indexed: %s", source)
            return doc_id
        doc = Document(id=doc_id, content=text, source=source, metadata=metadata or {})
        self._documents[doc_id] = doc
        chunks = self._chunk(doc)
        self._chunks.extend(chunks)
        log.info("Added document '%s' → %d chunks (id=%s)", source, len(chunks), doc_id)
        return doc_id

    def add_file(self, path: str) -> str:
        from pathlib import Path
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        return self.add_text(text, source=str(p), metadata={"filename": p.name, "extension": p.suffix})

    def add_directory(self, directory: str, extensions: list[str] | None = None) -> list[str]:
        from pathlib import Path
        extensions = extensions or [".txt", ".md"]
        ids = []
        for p in Path(directory).iterdir():
            if p.is_file() and p.suffix in extensions:
                ids.append(self.add_file(str(p)))
        return ids

    # ── Retrieval helpers ──────────────────────────────────────────────────

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        for c in self._chunks:
            if c.id == chunk_id:
                return c
        return None

    def stats(self) -> dict:
        return {
            "documents": len(self._documents),
            "chunks": len(self._chunks),
        }

    # ── Private ────────────────────────────────────────────────────────────

    @staticmethod
    def _make_id(text: str, source: str) -> str:
        return hashlib.sha256((source + text[:200]).encode()).hexdigest()[:16]

    def _chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content.strip()
        chunks = []
        idx = 0
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            # Try to break at sentence boundary
            if end < len(text):
                for sep in (".\n", ".\n", ". ", "\n\n", "\n"):
                    pos = text.rfind(sep, start, end)
                    if pos > start + CHUNK_SIZE // 2:
                        end = pos + len(sep)
                        break
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    id=f"{doc.id}_{idx}",
                    doc_id=doc.id,
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "source": doc.source,
                        "chunk_idx": idx,
                        "char_start": start,
                        "char_end": end,
                    },
                ))
                idx += 1
            start = end - CHUNK_OVERLAP
            if start >= len(text):
                break
        return chunks
