"""
Open-source dataset ingestion (Hugging Face `datasets`).

Instead of feeding dataset records straight into the vector index, records
are MATERIALISED as plain-text files under CORPUS_DIR/<dataset>/. This gives:

  - idempotent re-runs: each record gets a stable content-hash filename,
    so re-running the loader only writes NEW records
  - a single ingestion path: the index manager just syncs directories,
    whether documents came from a dataset, an upload, or a local file
  - easy inspection/debugging: every indexed document is a readable file

Datasets are streamed — nothing is fully downloaded into RAM — so even
Wikipedia-scale sources work with a bounded `max_docs`.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from config import CORPUS_DIR, DATASET_MAX_DOCS, DATASET_MIN_CHARS
from utils import get_logger

log = get_logger("DatasetLoader")

# Curated open-source datasets that work out of the box.
# Any other HF dataset can be used via load_hf_dataset(path=..., text_field=...).
PRESETS: dict[str, dict] = {
    "wikipedia-simple": {
        "path": "wikimedia/wikipedia",
        "config": "20231101.simple",
        "split": "train",
        "text_field": "text",
        "title_field": "title",
        "description": "Simple-English Wikipedia (~240k articles, clean prose)",
    },
    "wikipedia-en": {
        "path": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "text_field": "text",
        "title_field": "title",
        "description": "Full English Wikipedia (6M+ articles — use max_docs!)",
    },
    "ag-news": {
        "path": "fancyzhx/ag_news",
        "config": None,
        "split": "train",
        "text_field": "text",
        "title_field": None,
        "description": "AG News — 120k news articles in 4 categories",
    },
    "cc-news": {
        "path": "vblagoje/cc_news",
        "config": None,
        "split": "train",
        "text_field": "text",
        "title_field": "title",
        "description": "CC-News — 700k+ English news articles from CommonCrawl",
    },
    "squad": {
        "path": "rajpurkar/squad",
        "config": None,
        "split": "train",
        "text_field": "context",
        "title_field": "title",
        "description": "SQuAD reading-comprehension paragraphs (Wikipedia-based)",
    },
    "pubmed-summarization": {
        "path": "ccdv/pubmed-summarization",
        "config": None,
        "split": "train",
        "text_field": "article",
        "title_field": None,
        "description": "PubMed biomedical research articles",
    },
}


def list_presets() -> dict[str, str]:
    return {k: v["description"] for k, v in PRESETS.items()}


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()


def load_preset(name: str, max_docs: int = DATASET_MAX_DOCS) -> dict:
    if name not in PRESETS:
        raise ValueError(
            f"Unknown dataset preset '{name}'. Available: {', '.join(PRESETS)}. "
            "Or use load_hf_dataset() for any Hugging Face dataset."
        )
    p = PRESETS[name]
    return load_hf_dataset(
        path=p["path"], config_name=p["config"], split=p["split"],
        text_field=p["text_field"], title_field=p["title_field"],
        max_docs=max_docs, corpus_subdir=name,
    )


def load_hf_dataset(
    path: str,
    text_field: str,
    config_name: str | None = None,
    split: str = "train",
    title_field: str | None = None,
    max_docs: int = DATASET_MAX_DOCS,
    min_chars: int = DATASET_MIN_CHARS,
    corpus_subdir: str | None = None,
) -> dict:
    """Stream any HF dataset and materialise up to `max_docs` records as
    text files under CORPUS_DIR. Returns stats. Idempotent: existing files
    (same content hash) are skipped, so repeated runs pick up only new data.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The 'datasets' package is required for dataset ingestion. "
            "Run: pip install datasets"
        ) from e

    out_dir = CORPUS_DIR / _safe_name(corpus_subdir or path)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Streaming dataset %s (config=%s split=%s) → %s (max_docs=%d)",
             path, config_name, split, out_dir, max_docs)

    # Retry the initial connection — HF hub occasionally hiccups.
    ds = None
    for attempt in range(1, 4):
        try:
            ds = load_dataset(path, config_name, split=split, streaming=True)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise RuntimeError(f"Failed to open dataset '{path}': {e}") from e
            log.warning("Dataset load failed (attempt %d/3): %s — retrying...", attempt, e)
            time.sleep(3 * attempt)

    written = skipped = errors = seen = 0
    seen_hashes: set[str] = set()

    for record in ds:
        if written + skipped >= max_docs:
            break
        seen += 1
        try:
            text = (record.get(text_field) or "").strip()
            if len(text) < min_chars:
                continue
            title = (record.get(title_field) or "").strip() if title_field else ""
            content = f"{title}\n\n{text}" if title else text

            h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            if h in seen_hashes:            # in-run dedup (e.g. SQuAD contexts repeat)
                continue
            seen_hashes.add(h)

            fpath = out_dir / f"{h}.txt"
            if fpath.exists():
                skipped += 1
                continue
            fpath.write_text(content, encoding="utf-8")
            written += 1
            if written % 100 == 0:
                log.info("  ... %d documents written", written)
        except Exception as e:  # noqa: BLE001 — one bad record must not kill the run
            errors += 1
            if errors <= 5:
                log.warning("Skipping bad record #%d: %s", seen, e)

    stats = {
        "dataset": path,
        "corpus_dir": str(out_dir),
        "written": written,
        "skipped_existing": skipped,
        "errors": errors,
    }
    # Record provenance for the update job
    meta_path = out_dir / "_dataset.json"
    meta_path.write_text(json.dumps({
        "path": path, "config": config_name, "split": split,
        "text_field": text_field, "title_field": title_field,
        "last_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "last_stats": stats,
    }, indent=2), encoding="utf-8")

    log.info("Dataset ingestion done: %s", stats)
    return stats


def refresh_all(max_docs: int = DATASET_MAX_DOCS) -> list[dict]:
    """Re-pull every dataset previously materialised under CORPUS_DIR.
    Used by the scheduled update job — new records are appended, existing
    files are left untouched."""
    results = []
    for meta_path in sorted(CORPUS_DIR.glob("*/_dataset.json")):
        try:
            meta = json.loads(meta_path.read_text())
            results.append(load_hf_dataset(
                path=meta["path"], config_name=meta.get("config"),
                split=meta.get("split", "train"),
                text_field=meta["text_field"], title_field=meta.get("title_field"),
                max_docs=max_docs, corpus_subdir=meta_path.parent.name,
            ))
        except Exception as e:  # noqa: BLE001 — refresh one, fail one
            log.error("Refresh failed for %s: %s", meta_path.parent.name, e)
            results.append({"dataset": meta_path.parent.name, "error": str(e)})
    if not results:
        log.info("No previously ingested datasets found under %s", CORPUS_DIR)
    return results
