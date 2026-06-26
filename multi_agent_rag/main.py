"""
Production Multi-Agent RAG System
──────────────────────────────────
Pipeline (powered by LangGraph):
  QueryAnalyzer → Retriever (FAISS) → Analyzer → Synthesizer → Critic
                                                         ↑          │
                                                         └──────────┘
                                                      (revision loop)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag import DocumentStore, Ingestion, VectorRetriever
from agents import run_pipeline
from config import DATA_DIR, VECTOR_BACKEND
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


def build_knowledge_base() -> tuple[DocumentStore, VectorRetriever]:
    store = DocumentStore()
    ingestion = Ingestion(store)

    log.info("Ingesting documents from '%s'...", DATA_DIR)
    count = ingestion.ingest_directory(str(DATA_DIR), extensions=[".txt", ".md"])
    if count == 0:
        log.warning("No documents ingested — check DATA_DIR: %s", DATA_DIR)

    retriever = VectorRetriever(store, backend=VECTOR_BACKEND)
    retriever.build_index()
    return store, retriever


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


def main():
    print("=" * 70)
    print("  Production Multi-Agent RAG System  (LangGraph + FAISS + Claude)")
    print("=" * 70)

    _, retriever = build_knowledge_base()

    print(f"\nReady! {len(DEMO_QUERIES)} demo queries available.\n")
    for i, q in enumerate(DEMO_QUERIES, 1):
        print(f"  [{i}] {q}")
    print(f"  [0] Enter your own query")
    print(f"  [q] Quit")

    while True:
        choice = input("\nSelect query (0-{} or q): ".format(len(DEMO_QUERIES))).strip()
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

        result = run_pipeline(query, retriever)
        print_result(result)


if __name__ == "__main__":
    main()
