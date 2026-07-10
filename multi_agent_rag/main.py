"""
Production Multi-Agent RAG System
──────────────────────────────────
Pipeline (powered by LangGraph):
  QueryAnalyzer → Retriever (FAISS) → Analyzer → Synthesizer → Critic
                                                         ↑          │
                                                         └──────────┘
                                                      (revision loop)

Usage:
  python main.py                                   # interactive CLI
  python main.py --query "How does CRISPR work?"   # one-shot query
  python main.py --dataset wikipedia-simple --max-docs 300   # ingest first
  python main.py --check                           # provider + index health check
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag import IndexManager
from agents import run_pipeline
from llm import check_provider, LLMError
import config
from utils import get_logger

log = get_logger("Main")

DEMO_QUERIES = [
    "How does Retrieval Augmented Generation (RAG) work and what makes it better than a plain LLM?",
    "What are the main differences between CRISPR base editing and prime editing?",
    "Compare FAISS and ChromaDB as vector databases — when should I use each?",
    "What are the biggest risks of quantum computers for cybersecurity?",
    "How does the Artemis program plan to return humans to the Moon?",
    "Explain the economics of the global renewable energy transition.",
]


def print_result(result: dict):
    print("\n" + "=" * 70)
    print(f"QUERY: {result['query']}")
    print("=" * 70)
    print(result["final_answer"])
    print("-" * 70)
    m = result["metrics"]
    print(f"  Verdict: {result['verdict']}  |  Score: {result['score']}/10  "
          f"|  Retrieved: {result['retrieved_count']} chunks  "
          f"|  Time: {m['total_seconds']}s  |  Revisions: {m['revision_cycles']}")
    print("=" * 70 + "\n")


def interactive_loop(manager: IndexManager):
    print(f"\nReady! {len(DEMO_QUERIES)} demo queries available.\n")
    for i, q in enumerate(DEMO_QUERIES, 1):
        print(f"  [{i}] {q}")
    print("  [0] Enter your own query")
    print("  [q] Quit")

    while True:
        try:
            choice = input(f"\nSelect query (0-{len(DEMO_QUERIES)} or q): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice.lower() == "q":
            break
        if choice == "0":
            query = input("Enter your query: ").strip()
        elif choice.isdigit() and 1 <= int(choice) <= len(DEMO_QUERIES):
            query = DEMO_QUERIES[int(choice) - 1]
        else:
            print("Invalid choice.")
            continue
        if not query:
            continue

        try:
            result = run_pipeline(query, manager.retriever)
            print_result(result)
        except LLMError as e:
            log.error("Pipeline failed: %s", e)
            print(f"\n  ✗ LLM error: {e}\n  Check your provider with: python ingest_cli.py check\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Production Multi-Agent RAG")
    parser.add_argument("--query", help="Run one query and exit (non-interactive)")
    parser.add_argument("--dataset", help="Ingest an open-source dataset preset first "
                                          "(e.g. wikipedia-simple, ag-news, squad)")
    parser.add_argument("--max-docs", type=int, default=config.DATASET_MAX_DOCS)
    parser.add_argument("--backend", default=config.VECTOR_BACKEND,
                        choices=["faiss", "chroma", "numpy"])
    parser.add_argument("--force-reindex", action="store_true",
                        help="Ignore the embedding cache and re-embed everything")
    parser.add_argument("--check", action="store_true",
                        help="Health-check the LLM provider and index, then exit")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Production Multi-Agent RAG  "
          f"(LangGraph + {args.backend.upper()} + {config.LLM_PROVIDER})")
    print("=" * 70)

    if args.check:
        status = check_provider()
        print(f"\nProvider: {status['provider']} ({status['model']}) — "
              f"{'OK' if status['ok'] else 'FAILED'}: {status['detail']}\n")
        return 0 if status["ok"] else 1

    # Fail fast with a clear message if the provider is misconfigured
    status = check_provider()
    if not status["ok"]:
        log.error("LLM provider not ready: %s", status["detail"])
        print(f"\n  ✗ {status['detail']}\n")
        return 1

    if args.dataset:
        from rag.dataset_loader import load_preset
        try:
            stats = load_preset(args.dataset, max_docs=args.max_docs)
            print(f"\nDataset ingested: {stats['written']} new, "
                  f"{stats['skipped_existing']} already present\n")
        except Exception as e:  # noqa: BLE001 — CLI boundary
            log.error("Dataset ingestion failed: %s", e)
            return 1

    manager = IndexManager(backend=args.backend)
    report = manager.sync(force=args.force_reindex)
    if report["chunks"] == 0:
        log.error("Knowledge base is empty. Add files to %s or run: "
                  "python ingest_cli.py dataset --name wikipedia-simple", config.DATA_DIR)
        return 1

    if args.query:
        try:
            result = run_pipeline(args.query, manager.retriever)
        except LLMError as e:
            log.error("Pipeline failed: %s", e)
            return 1
        print_result(result)
        return 0

    interactive_loop(manager)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
