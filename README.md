# Production Multi-Agent RAG System

A production-deployable **multi-agent Retrieval Augmented Generation** system:

- **LangGraph** — stateful, cyclic agent orchestration (automatic revision loop)
- **5 specialized agents** — QueryAnalyzer → Retriever → Analyzer → Synthesizer → Critic
- **4 LLM providers, one env var** — Anthropic **Claude**, local **Ollama**, local **LM Studio**, or any OpenAI-compatible endpoint (vLLM, Groq, Together, …)
- **FAISS / ChromaDB / NumPy** vector backends
- **Open-source datasets** — stream Wikipedia, AG News, CC-News, SQuAD, PubMed (or any Hugging Face dataset) straight into the knowledge base
- **Regular data injection & updates** — incremental indexing with an embedding cache, cron/Docker scheduler, zero-downtime index hot-swap
- **Web UI + REST API + Docker** — built-in browser UI, FastAPI server with Swagger docs, health checks, compose deployment

> 📖 **Full system documentation** (how every part works, ops, scaling, troubleshooting): [DOCUMENTATION.md](DOCUMENTATION.md)
>
> Reference articles: [Production Multi-Agent RAG](https://sabbirahmedsaqlain.github.io/articles/production-multi-agent-rag/) · [Embeddings in RAG](https://sabbirahmedsaqlain.github.io/articles/embeddings-in-rag-complete-practical-guide/) · [Vector Databases](https://sabbirahmedsaqlain.github.io/articles/vector-database/) · [LangChain & LangGraph](https://sabbirahmedsaqlain.github.io/articles/langchain-langgraph/) · [RAG Evaluation](https://sabbirahmedsaqlain.github.io/articles/rag-performance-evaluation-guide/)

---

## Quick Start (3 commands)

```bash
./scripts/setup.sh          # venv + dependencies + .env template + provider check
# edit .env → pick your LLM provider (see below)
./scripts/run.sh            # interactive CLI on the bundled sample data
```

Everything also works via `make`: `make setup`, `make run`, `make serve`, `make ingest`, `make update`, `make check`, `make docker`.

### Choosing an LLM provider

Set `LLM_PROVIDER` in `.env` — nothing else changes:

| Provider | Setup | `.env` |
|---|---|---|
| **Claude (Anthropic)** — best quality | get an API key | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=sk-ant-...` |
| **Ollama** — free, local, open-source | [install](https://ollama.com), `ollama pull llama3.1:8b` | `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=llama3.1:8b` |
| **LM Studio** — free, local, GUI | [install](https://lmstudio.ai), load a model, enable *Local Server* | `LLM_PROVIDER=lmstudio` (model auto-detected) |
| **Any OpenAI-compatible** — vLLM, llama.cpp, Groq, Together, OpenAI | have the endpoint URL + key | `LLM_PROVIDER=openai`, `OPENAI_BASE_URL=...`, `OPENAI_MODEL=...` |

Verify your setup any time:

```bash
./scripts/check.sh           # probes Ollama/LM Studio, checks provider + index
```

### Loading open-source data

```bash
./scripts/ingest.sh list                                   # show dataset presets
./scripts/ingest.sh --name wikipedia-simple --max-docs 500 # Simple-English Wikipedia
./scripts/ingest.sh --name ag-news --max-docs 200          # news articles
# any Hugging Face dataset:
./scripts/ingest.sh --hf-path cnn_dailymail --hf-config 3.0.0 --text-field article
```

Datasets are streamed (never fully downloaded), deduplicated, and materialised as
files under `multi_agent_rag/corpus/` — re-running only adds *new* records.
You can also just drop `.txt`/`.md` files into `multi_agent_rag/data/`.

### Running queries

```bash
./scripts/run.sh                                          # interactive menu
./scripts/run.sh --query "How does CRISPR work?"          # one-shot
./scripts/run.sh --dataset squad --max-docs 300           # ingest then run
./scripts/run.sh --backend chroma                         # persistent vector store
```

### Web UI + Production API server

```bash
./scripts/serve.sh
```

Then open:

| URL | What |
|---|---|
| `http://localhost:8000/` | **Web UI** — ask questions, see verdict/score/timings, ingest datasets, monitor health |
| `http://localhost:8000/docs` | **Swagger UI** — interactive API docs, try every endpoint |
| `http://localhost:8000/redoc` | ReDoc — reference-style API docs |

```bash
curl -s -X POST localhost:8000/query -H 'Content-Type: application/json' \
     -d '{"query": "Compare FAISS and ChromaDB"}' | jq -r .final_answer

curl -s localhost:8000/health          # provider + index status
curl -s -X POST localhost:8000/ingest/dataset -H 'Content-Type: application/json' \
     -d '{"name": "ag-news", "max_docs": 200}'      # background ingest + re-index
```

### Docker deployment

```bash
cp .env.example .env   # configure provider
docker compose up -d --build                     # API only
docker compose --profile ollama up -d --build    # + local open-source LLM
docker compose exec ollama ollama pull llama3.1:8b
docker compose --profile scheduler up -d         # + hourly automatic data updates
```

### Regular data updates

The knowledge base stays fresh through a **pull → incremental sync → hot-swap** loop
(only new documents get embedded, thanks to a per-chunk embedding cache; a running
API swaps its index atomically with zero downtime):

```bash
./scripts/update_data.sh      # one manual update (lock-protected, cron-safe)

# hourly via cron:
# 0 * * * * /path/to/repo/scripts/update_data.sh >> /path/to/repo/multi_agent_rag/logs/update.log 2>&1

# or in Docker: the --profile scheduler service does this automatically
# or on demand against a running API:
curl -X POST localhost:8000/refresh -H 'Content-Type: application/json' -d '{}'
```

Details in [DOCUMENTATION.md §4.5](DOCUMENTATION.md).

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
                                          │
                        every agent calls llm.chat() — provider-agnostic:
                        anthropic │ ollama │ lmstudio │ openai-compatible
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

At query time the query is embedded with the same model and compared by cosine
similarity against all stored vectors; the most similar chunks are fed to the LLM
as context.

### FAISS vs ChromaDB vs NumPy

| | **NumPy** | **FAISS** | **ChromaDB** |
|---|---|---|---|
| Storage | RAM | RAM | Disk (persistent) |
| Search type | Brute-force exact | Exact (IndexFlatIP) | Approximate (HNSW) |
| Scale | ~10K chunks | Millions | Millions |
| Persistence | ❌ Lost on restart | ❌ (fast rebuild from embedding cache) | ✅ Survives restart |
| **When to use** | Prototyping | Production in-memory | Production persistent |

**This system uses FAISS by default.** Switch with `VECTOR_BACKEND=chroma`.

---

## The 5 Agents

### 1. QueryAnalyzerAgent
Analyses the user query before retrieval: intent classification plus 2–3 diverse
**query rephrasings** (multi-query expansion). Each rephrasing is retrieved
independently, then deduplicated — dramatically improving recall for ambiguous queries.

### 2. RetrieverAgent
Executes vector search for each rephrased query, deduplicates results, applies a
score threshold, and runs an LLM-based relevance filtering pass.

### 3. AnalyzerAgent
Deep-reads the retrieved chunks to extract key facts with source attribution,
supporting quotes, contradictions between sources, and information gaps.

### 4. SynthesizerAgent
Writes a well-structured, source-grounded answer. On revision cycles, applies the
critic's specific feedback to improve the draft.

### 5. CriticAgent
Evaluates the draft on accuracy, completeness, clarity, citations, and conciseness.
Produces a VERDICT (APPROVED / NEEDS_REVISION) with a score out of 10. On
NEEDS_REVISION the graph loops back to the synthesizer, up to `MAX_REVISION_CYCLES`.

---

## Project Structure

```
Multi_Agent_RAG/
├── README.md                        # this file — how to run
├── DOCUMENTATION.md                 # full system documentation
├── Makefile                         # make setup / run / serve / ingest / update / docker
├── Dockerfile                       # production image (healthcheck, non-root, prebaked model)
├── docker-compose.yml               # api + optional ollama + optional data-updater
├── .env.example                     # annotated config template → copy to .env
├── scripts/
│   ├── setup.sh                     # one-time setup (idempotent, error-handled)
│   ├── run.sh                       # CLI runner
│   ├── serve.sh                     # API server (uvicorn)
│   ├── ingest.sh                    # open-source dataset ingestion
│   ├── update_data.sh               # cron-safe regular data update job
│   ├── check.sh                     # provider + index health check
│   └── lib.sh                       # shared bash helpers
└── multi_agent_rag/
    ├── main.py                      # interactive / one-shot CLI
    ├── api.py                       # FastAPI production server (UI + Swagger + REST)
    ├── static/index.html            # built-in web UI (single file, no build step)
    ├── ingest_cli.py                # data management CLI (dataset/refresh/sync/check)
    ├── config.py                    # all configuration (env-overridable)
    ├── requirements.txt
    ├── llm/
    │   └── providers.py             # anthropic / ollama / lmstudio / openai-compatible + retries
    ├── agents/
    │   ├── base_agent.py            # provider-agnostic LLM access
    │   ├── graph.py                 # LangGraph state machine & routing
    │   ├── query_analyzer.py        # intent + multi-query expansion
    │   ├── retriever_agent.py       # multi-query retrieval + dedup
    │   ├── analyzer_agent.py        # fact extraction & gap analysis
    │   ├── synthesizer_agent.py     # answer composition & revision
    │   └── critic_agent.py          # quality scoring & validation
    ├── rag/
    │   ├── document_store.py        # documents + sentence-aware chunking
    │   ├── ingestion.py             # file loading & cleaning
    │   ├── dataset_loader.py        # Hugging Face open-source datasets → corpus/
    │   ├── index_manager.py         # incremental sync, manifest, embedding cache
    │   └── retriever.py             # FAISS / ChromaDB / NumPy backends
    ├── utils/                       # logging + per-step metrics
    ├── data/                        # hand-curated knowledge base (8 sample topics)
    ├── corpus/                      # auto: materialised dataset documents
    ├── index_state/                 # auto: manifest.json + embeddings.npz cache
    ├── logs/                        # auto: rag_system.log
    └── chroma_db/                   # auto: if VECTOR_BACKEND=chroma
```

---

## Programmatic API

```python
import sys; sys.path.insert(0, "multi_agent_rag")
from rag import IndexManager
from agents import run_pipeline

manager = IndexManager()          # backend="faiss" | "chroma" | "numpy"
manager.sync()                    # incremental: only new chunks get embedded

result = run_pipeline("How does CRISPR work?", manager.retriever)
print(result["final_answer"])
print(f"Score: {result['score']}/10  Verdict: {result['verdict']}")
print(f"Time: {result['metrics']['total_seconds']}s  "
      f"Revisions: {result['metrics']['revision_cycles']}")
```

Ingest an open-source dataset programmatically:

```python
from rag.dataset_loader import load_preset, load_hf_dataset
load_preset("wikipedia-simple", max_docs=500)
load_hf_dataset(path="cnn_dailymail", config_name="3.0.0",
                text_field="article", max_docs=300)
manager.sync()                    # picks up the new documents
```

---

## Configuration

Everything is set via `.env` (see [.env.example](.env.example)) or environment
variables. The most important ones:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `ollama` \| `lmstudio` \| `openai` |
| `CLAUDE_MODEL` | `claude-opus-4-8` | model when using Anthropic |
| `OLLAMA_MODEL` | `llama3.1:8b` | model when using Ollama |
| `VECTOR_BACKEND` | `faiss` | `faiss` \| `chroma` \| `numpy` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `600` / `80` | chunking parameters |
| `TOP_K_RETRIEVE` / `TOP_K_FINAL` | `8` / `5` | retrieval depth |
| `MAX_REVISION_CYCLES` | `2` | critic → synthesizer loop bound |
| `DATASET_MAX_DOCS` | `500` | default cap for dataset ingestion |
| `LLM_MAX_RETRIES` / `LLM_TIMEOUT` | `3` / `300` | resilience settings |

Full reference: [DOCUMENTATION.md §6](DOCUMENTATION.md).

---

## Adding Your Own Documents

Drop `.txt` or `.md` files into `multi_agent_rag/data/` and run
`./scripts/run.sh` (the index syncs on start), or against a running API:

```bash
curl -X POST localhost:8000/ingest/text -H 'Content-Type: application/json' \
     -d '{"content": "Your document text...", "source": "my-notes"}'
```

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

Then wire it in `agents/graph.py`:

```python
graph.add_node("fact_check", lambda s: node_fact_check(s, agents))
graph.add_edge("synthesis", "fact_check")
graph.add_edge("fact_check", "critic")
```

Because agents talk to the LLM only through `self._call`, a new agent
automatically works on all four providers.

---

## Troubleshooting (quick)

| Problem | Fix |
|---|---|
| Provider errors on start | `./scripts/check.sh` tells you exactly what's wrong |
| Ollama: "no loaded models" | `ollama pull llama3.1:8b` |
| First run slow | one-time downloads (torch, embedding model); later runs hit the cache |
| New data not in answers | `./scripts/update_data.sh` or `curl -X POST :8000/refresh` |
| Changed embedding model | `python multi_agent_rag/ingest_cli.py sync --force` |

More in [DOCUMENTATION.md §9](DOCUMENTATION.md).

## License

MIT — free to use, modify, and distribute.
