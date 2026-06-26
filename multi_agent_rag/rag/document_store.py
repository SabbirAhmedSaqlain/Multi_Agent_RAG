import os
import json
from dataclasses import dataclass, field
from typing import Optional
from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Document:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    source: str = ""


@dataclass
class Chunk:
    id: str
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list] = None


class DocumentStore:
    def __init__(self):
        self.documents: dict[str, Document] = {}
        self.chunks: list[Chunk] = []

    def add_document(self, content: str, source: str = "", metadata: dict | None = None) -> str:
        doc_id = f"doc_{len(self.documents)}"
        doc = Document(id=doc_id, content=content, source=source, metadata=metadata or {})
        self.documents[doc_id] = doc
        self._chunk_document(doc)
        return doc_id

    def add_from_file(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return self.add_document(content, source=filepath, metadata={"filename": os.path.basename(filepath)})

    def load_from_directory(self, directory: str, extensions: list[str] | None = None) -> list[str]:
        extensions = extensions or [".txt", ".md", ".json"]
        doc_ids = []
        for fname in os.listdir(directory):
            if any(fname.endswith(ext) for ext in extensions):
                doc_ids.append(self.add_from_file(os.path.join(directory, fname)))
        return doc_ids

    def _chunk_document(self, doc: Document):
        text = doc.content
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end]
            chunk = Chunk(
                id=f"{doc.id}_chunk_{chunk_idx}",
                doc_id=doc.id,
                content=chunk_text,
                metadata={**doc.metadata, "source": doc.source, "chunk_idx": chunk_idx},
            )
            self.chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
            chunk_idx += 1

    def get_all_chunks(self) -> list[Chunk]:
        return self.chunks

    def stats(self) -> dict:
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
        }
