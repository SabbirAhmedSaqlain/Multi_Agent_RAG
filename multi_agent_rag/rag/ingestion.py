from __future__ import annotations
"""
Production ingestion pipeline:
  - Load files from a directory
  - Clean and preprocess text
  - Add to document store
  - Build / update the vector index
"""
from pathlib import Path
from .document_store import DocumentStore
from utils import get_logger

log = get_logger("Ingestion")


class Ingestion:
    def __init__(self, store: DocumentStore):
        self.store = store

    def ingest_directory(
        self,
        directory: str,
        extensions: list[str] | None = None,
        recursive: bool = False,
    ) -> int:
        extensions = extensions or [".txt", ".md"]
        root = Path(directory)
        if not root.exists():
            log.warning("Directory does not exist: %s", directory)
            return 0

        pattern = "**/*" if recursive else "*"
        files = [p for p in root.glob(pattern) if p.is_file() and p.suffix in extensions]
        log.info("Found %d files to ingest from '%s'", len(files), directory)

        count = 0
        for fpath in files:
            try:
                raw = fpath.read_text(encoding="utf-8", errors="replace")
                cleaned = self._clean(raw)
                self.store.add_text(cleaned, source=str(fpath), metadata={
                    "filename": fpath.name,
                    "extension": fpath.suffix,
                    "size_bytes": fpath.stat().st_size,
                })
                count += 1
            except Exception as e:
                log.error("Failed to ingest '%s': %s", fpath, e)

        log.info("Ingested %d/%d files → %s", count, len(files), self.store.stats())
        return count

    def ingest_texts(self, texts: list[dict]) -> int:
        """texts: list of {"content": str, "source": str, "metadata": dict}"""
        for item in texts:
            self.store.add_text(
                item["content"],
                source=item.get("source", "inline"),
                metadata=item.get("metadata", {}),
            )
        return len(texts)

    @staticmethod
    def _clean(text: str) -> str:
        import re
        # Collapse excessive whitespace while preserving paragraph breaks
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\t", " ", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
