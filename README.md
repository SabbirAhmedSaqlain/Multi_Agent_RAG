# Production Multi-Agent RAG System

A production-grade **Retrieval Augmented Generation (RAG)** system built with:

- **LangGraph** — stateful, cyclic agent orchestration (with automatic revision loop)
- **FAISS** — fast in-memory vector search (exact cosine via IndexFlatIP)
- **ChromaDB** — persistent disk-backed vector store (HNSW approximate nearest-neighbor)
- **Claude Opus 4.8** — all agents use adaptive thinking for best reasoning quality
- **5 specialized agents** — each with a distinct responsibility, passing structured data through a typed state graph

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │         LangGraph State Machine          │
                        │                                          │
  User Query ──────────►│  [1] QueryAnalyzer                       │
                        │      ↓                                   │
                        │  [2] RetrieverAgent ◄── FAISS / ChromaDB │
                        │      ↓                                   │
                        │  [3] AnalyzerAgent                       │
                        │      ↓                                   │
                        │  [4] SynthesizerAgent ◄─────────────┐   │
                        │      ↓                               │   │
                        │  [5] CriticAgent ── NEEDS_REVISION ──┘   │
                        │      │                                   │
                        │      └── APPROVED ──► Final Answer       │
                        └─────────────────────────────────────────┘
```

### Why LangGraph?

LangGraph models the pipeline as a **directed graph with conditional edges** — not a sequential function call chain. This means:

- **Cycles are first-class**: the critic can send control back to the synthesizer for revision without any special-case code
- **State is typed and shared**: every node reads/writes a single `RAGState` TypedDict — no function argument soup
- **Easy to extend**: add a new node, wire it with `add_edge`, done
- **Checkpointing ready**: LangGraph supports mid-graph state persistence for long-running pipelines

---

## What is a Vector Database?

A **vector database** stores data as **numerical vectors** (embeddings) rather than rows/columns. Each document chunk is converted into a dense vector (e.g. 384 floats) that captures its *meaning*. Search is done by mathematical similarity — not keyword matching.

```
Text: "FAISS enables fast nearest-neighbor search"
         │
         ▼  (SentenceTransformer embedding model)
Vector: [0.21, -0.09, 0.84, 0.13, ...]   ← 384 numbers representing meaning
```

At query time:
```
"How to search embeddings quickly?"
         │
         ▼  (same embedding model)
[0.19, -0.11, 0.81, ...]                  ← query vector

         │  cosine similarity comparison against all stored vectors
         ▼
Most similar chunks returned → fed to the LLM as context
```

### How Data is Stored in This System

```
Raw Text File
    │
    ▼
Ingestion Pipeline (cleaning, chunking with overlap)
    │
    ├── Chunk 0: "FAISS is a library by Meta..."
    ├── Chunk 1: "...uses IVF and HNSW indices..."
    └── Chunk 2: "...exact vs approximate search..."
         │
         ▼
SentenceTransformer ("all-MiniLM-L6-v2")
         │
         ▼
┌─── FAISS Backend ──────────────────────────────┐
│  IndexFlatIP (Inner Product on unit vectors)   │
│  All vectors stored in RAM                     │
│  Exact cosine similarity search                │
│  ~1ms search over 100K vectors                 │
└────────────────────────────────────────────────┘

┌─── ChromaDB Backend ───────────────────────────┐
│  HNSW (Hierarchical Navigable Small World)     │
│  Vectors persisted to disk (./chroma_db/)      │
│  Approximate nearest-neighbor search           │
│  Survives process restarts                     │
└────────────────────────────────────────────────┘
```

### FAISS vs ChromaDB vs NumPy

| | **NumPy** | **FAISS** | **ChromaDB** |
|---|---|---|---|
| Storage | RAM | RAM | Disk (persistent) |
| Search type | Brute-force exact | Exact (IndexFlatIP) | Approximate (HNSW) |
| Scale | ~10K chunks | Millions | Millions |
| Persistence | ❌ Lost on restart | ❌ Lost on restart | ✅ Survives restart |
| Setup | Zero | `faiss-cpu` | `chromadb` |
| Speed (1M vectors) | Seconds | Milliseconds | Milliseconds |
| **When to use** | Prototyping | Production in-memory | Production persistent |

**This system uses FAISS by default** (fast, exact, production-grade). Switch to ChromaDB with `VECTOR_BACKEND=chroma`.

---

## The 5 Agents

### 1. QueryAnalyzerAgent
Analyses the user query before retrieval to improve search quality.

**Output:**
```json
{
  "intent": "comparative",
  "search_queries": [
    "FAISS vector database performance",
    "ChromaDB persistent vector store",
    "approximate vs exact nearest neighbor search"
  ],
  "key_concepts": ["FAISS", "ChromaDB", "vector database"],
  "requires_detail": true
}
```
Multiple rephrasings are retrieved independently, then deduplicated — dramatically improving recall for ambiguous queries.

### 2. RetrieverAgent
Executes vector search for each rephrased query using FAISS/ChromaDB, deduplicates results, applies a score threshold, and runs an LLM-based relevance filtering pass.

### 3. AnalyzerAgent
Deep-reads the retrieved chunks to extract:
- Key facts with source attribution
- Supporting evidence (direct quotes)
- Contradictions between sources
- Information gaps (what the query asks but sources don't answer)

### 4. SynthesizerAgent
Writes a well-structured, source-grounded answer. On revision cycles, applies the critic's specific feedback to improve the draft.

### 5. CriticAgent
Evaluates the draft on five dimensions: accuracy, completeness, clarity, citations, conciseness. Produces a VERDICT (APPROVED / NEEDS_REVISION) with a score out of 10. If revision is needed and the iteration limit hasn't been reached, the graph loops back to the synthesizer.

---

## Project Structure

```
MultiAgent/
├── README.md
└── multi_agent_rag/
    ├── main.py                      # Interactive CLI demo
    ├── config.py                    # All configuration in one place
    ├── requirements.txt
    ├── agents/
    │   ├── __init__.py
    │   ├── base_agent.py            # Claude client, streaming, logging
    │   ├── graph.py                 # LangGraph state machine & routing
    │   ├── query_analyzer.py        # Query intent + multi-query expansion
    │   ├── retriever_agent.py       # Multi-query retrieval + dedup
    │   ├── analyzer_agent.py        # Fact extraction & gap analysis
    │   ├── synthesizer_agent.py     # Answer composition & revision
    │   └── critic_agent.py          # Quality scoring & validation
    ├── rag/
    │   ├── __init__.py
    │   ├── document_store.py        # Document + chunk storage
    │   ├── ingestion.py             # File loading, cleaning, chunking
    │   └── retriever.py             # FAISS / ChromaDB / NumPy backends
    ├── utils/
    │   ├── logger.py                # Structured logging (file + console)
    │   └── metrics.py               # Per-step timing & revision tracking
    ├── data/                        # Knowledge base documents
    │   ├── artificial_intelligence.txt
    │   ├── large_language_models.txt
    │   ├── quantum_computing.txt
    │   ├── renewable_energy.txt
    │   ├── climate_change.txt
    │   ├── biotechnology.txt
    │   ├── space_exploration.txt
    │   └── economics_finance.txt
    ├── logs/                        # Auto-created: rag_system.log
    └── chroma_db/                   # Auto-created if using ChromaDB backend
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Anthropic API key

### Install

```bash
cd MultiAgent
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r multi_agent_rag/requirements.txt
```

### Configure

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: choose vector backend (default: faiss)
export VECTOR_BACKEND=faiss       # faiss | chroma | numpy

# Optional: log verbosity
export LOG_LEVEL=INFO             # DEBUG | INFO | WARNING
```

### Run

```bash
cd multi_agent_rag
python main.py
```

---

## Usage

### Interactive CLI

```
$ python main.py

  Production Multi-Agent RAG System  (LangGraph + FAISS + Claude)

  [1] How does RAG work and what makes it better than a plain LLM?
  [2] What are the main differences between CRISPR base editing and prime editing?
  [3] Compare FAISS and ChromaDB — when should I use each?
  ...
  [0] Enter your own query

Select query: 1
```

### Programmatic API

```python
import sys; sys.path.insert(0, "multi_agent_rag")
from rag import DocumentStore, Ingestion, VectorRetriever
from agents import run_pipeline

# 1. Build knowledge base
store = DocumentStore()
ingestion = Ingestion(store)
ingestion.ingest_directory("./data")          # or any directory
# ingestion.ingest_texts([{"content": "...", "source": "doc1"}])

# 2. Build vector index
retriever = VectorRetriever(store, backend="faiss")
retriever.build_index()

# 3. Run the full multi-agent pipeline
result = run_pipeline("How does CRISPR work?", retriever)

print(result["final_answer"])
print(f"Score: {result['score']}/10")
print(f"Time: {result['metrics']['total_seconds']}s")
print(f"Revisions: {result['metrics']['revision_cycles']}")
```

### Result Object

```python
{
  "query": "How does CRISPR work?",
  "final_answer": "...",          # final validated answer text
  "verdict": "APPROVED",          # or "NEEDS_REVISION" (if max cycles reached)
  "score": 9,                     # critic score 1-10
  "retrieved_count": 5,           # chunks used as context
  "metrics": {
    "total_seconds": 42.1,
    "revision_cycles": 1,
    "steps": {
      "query_analysis": 2.1,
      "retrieval": 1.4,
      "analysis": 8.7,
      "synthesis_iter0": 12.3,
      "critic_iter0": 6.2,
      "synthesis_iter1": 9.8,    # only if revision occurred
      "critic_iter1": 5.9
    }
  }
}
```

---

## Configuration Reference

`multi_agent_rag/config.py`:

| Variable | Default | Description |
|---|---|---|
| `CLAUDE_MODEL` | `claude-opus-4-8` | Model for all agents |
| `VECTOR_BACKEND` | `faiss` | `faiss` \| `chroma` \| `numpy` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `CHUNK_SIZE` | `600` | Characters per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between adjacent chunks |
| `TOP_K_RETRIEVE` | `8` | Chunks retrieved before filtering |
| `TOP_K_FINAL` | `5` | Chunks kept after score threshold |
| `SCORE_THRESHOLD` | `0.25` | Min cosine similarity to keep |
| `MAX_REVISION_CYCLES` | `2` | Max critic→synthesizer iterations |
| `MAX_TOKENS` | `8192` | Token budget per agent call |

---

## How the Revision Loop Works

```
Synthesizer produces draft
         │
         ▼
    CriticAgent
         │
    ┌────┴─────────────────────────────────┐
    │                                      │
 VERDICT: NEEDS_REVISION            VERDICT: APPROVED
 iteration < MAX_REVISION_CYCLES          │
    │                                     ▼
    │                              Return final answer
    ▼
SynthesizerAgent (revision mode)
  ← receives critic's REVISION_INSTRUCTIONS
  ← re-reads original context
  → produces improved draft
         │
         ▼
    CriticAgent (again)
         │
    ... (up to MAX_REVISION_CYCLES times)
```

This loop guarantees answer quality without infinite loops. Setting `MAX_REVISION_CYCLES=0` disables revision entirely.

---

## Adding Your Own Documents

Drop any `.txt` or `.md` files into `multi_agent_rag/data/` and restart. The ingestion pipeline automatically discovers and indexes them.

Or point to a different directory:

```python
ingestion.ingest_directory("/path/to/my/documents", extensions=[".txt", ".md", ".rst"])
```

---

## Adding a New Agent

```python
# agents/fact_checker.py
from .base_agent import BaseAgent

class FactCheckerAgent(BaseAgent):
    name = "FactChecker"

    def run(self, draft: str, sources: str) -> dict:
        result = self._call(
            "You are a fact-checker. Identify any claims not supported by sources.",
            f"Draft:\n{draft}\n\nSources:\n{sources}"
        )
        return {"fact_check": result}
```

Then add a node in `agents/graph.py`:

```python
graph.add_node("fact_check", lambda s: node_fact_check(s, agents))
graph.add_edge("synthesis", "fact_check")
graph.add_edge("fact_check", "critic")
```

---

## Included Knowledge Base

| File | Topics |
|---|---|
| `artificial_intelligence.txt` | History, ML algorithms, deep learning, LLMs, ethics |
| `large_language_models.txt` | Transformers, RLHF, RAG, multi-agent systems, benchmarks |
| `quantum_computing.txt` | Qubits, NISQ, Shor/Grover algorithms, hardware platforms |
| `renewable_energy.txt` | Solar, wind, hydro, storage, economics, policy |
| `climate_change.txt` | Science, projections, tipping points, mitigation |
| `biotechnology.txt` | CRISPR, mRNA, genomics, synthetic biology, precision medicine |
| `space_exploration.txt` | History, Mars missions, Artemis, commercial space, JWST |
| `economics_finance.txt` | Markets, crypto, trade, development economics, regulations |

---

## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Claude API (all agent LLM calls) |
| `langgraph` | Graph-based agent orchestration, conditional routing |
| `langchain-core` | Base primitives for langgraph |
| `faiss-cpu` | Fast exact cosine similarity search (FAISS IndexFlatIP) |
| `chromadb` | Persistent vector store with HNSW ANN index |
| `sentence-transformers` | Text → vector embeddings |
| `numpy` | Vector math (also the fallback backend) |
| `torch` | Required by sentence-transformers |

---

## License

MIT — free to use, modify, and distribute.
