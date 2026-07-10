# Multi-Agent RAG — Full System Documentation

This document explains **how everything works**: the architecture, every agent, the
data/ingestion pipeline, the LLM provider layer, the update process, the API, and
how to deploy and operate the system in production.

Companion reading (the design follows these articles):

- [Production-Level Multi-Agent RAG System](https://sabbirahmedsaqlain.github.io/articles/production-multi-agent-rag/)
- [Embeddings in RAG: A Complete Practical Guide](https://sabbirahmedsaqlain.github.io/articles/embeddings-in-rag-complete-practical-guide/)
- [Production-Level Vector Database](https://sabbirahmedsaqlain.github.io/articles/vector-database/)
- [LangChain and LangGraph: Beginner to Production](https://sabbirahmedsaqlain.github.io/articles/langchain-langgraph/)
- [How to Validate and Improve RAG Performance](https://sabbirahmedsaqlain.github.io/articles/rag-performance-evaluation-guide/)

---

## 1. System Overview

```
                 ┌──────────────────────── Data plane ────────────────────────┐
                 │                                                            │
 HF open-source  │  dataset_loader ──► corpus/<dataset>/*.txt   (materialised)│
 datasets ──────►│                                                            │
                 │  data/*.txt  (hand-curated files)                          │
                 │        │                                                   │
                 │        ▼                                                   │
                 │  IndexManager.sync()                                       │
                 │    ├─ manifest.json      (file-hash change detection)      │
                 │    ├─ embeddings.npz     (per-chunk embedding cache)       │
                 │    └─ VectorRetriever    (FAISS / ChromaDB / NumPy)        │
                 └────────────────────────────┬───────────────────────────────┘
                                              │ retrieve(query)
                 ┌──────────────────────── Agent plane ───────────────────────┐
                 │              LangGraph state machine                       │
  User query ───►│  QueryAnalyzer → Retriever → Analyzer → Synthesizer        │
                 │                                              │  ▲          │
                 │                                              ▼  │revision  │
                 │                                           Critic ──────────┼──► Answer
                 └────────────────────────────────────────────────────────────┘
                                              │ every agent calls
                 ┌──────────────────────── LLM plane ─────────────────────────┐
                 │  llm.chat()  — one interface, four providers               │
                 │  anthropic │ ollama (local) │ lmstudio (local) │ openai-*  │
                 │  retry + exponential backoff + fail-fast config errors     │
                 └────────────────────────────────────────────────────────────┘
```

Three independent planes:

| Plane | Code | Responsibility |
|---|---|---|
| **Data** | `rag/` | Ingest documents (files + open-source datasets), chunk, embed, index, keep updated |
| **Agents** | `agents/` | The 5-agent LangGraph pipeline that turns a query + retrieved context into a validated answer |
| **LLM** | `llm/` | Provider-agnostic model access with retries; swap Claude ↔ local models with one env var |

Entry points:

| Entry point | File | Use |
|---|---|---|
| Interactive / one-shot CLI | `multi_agent_rag/main.py` | development, demos, scripting |
| REST API | `multi_agent_rag/api.py` | production serving |
| Data management CLI | `multi_agent_rag/ingest_cli.py` | ingestion, refresh, health checks (cron-friendly) |

---

## 2. The Agent Pipeline (LangGraph)

The pipeline is a **directed graph with a cycle**, not a call chain. All nodes read and
write one typed `RAGState` (`agents/graph.py`). The cycle implements automatic answer
revision.

```
query_analysis → retrieval → analysis → synthesis → critic
                                            ▲           │
                                            └───────────┘
                              NEEDS_REVISION and iteration ≤ MAX_REVISION_CYCLES
```

### 2.1 QueryAnalyzerAgent (`agents/query_analyzer.py`)
Runs *before* retrieval. Produces JSON: intent classification, 2–3 diverse
**query rephrasings** (multi-query expansion), key concepts, and a detail flag.
Each rephrasing is retrieved independently and results are deduplicated — this
is the single biggest recall improvement for ambiguous queries.
If the LLM returns malformed JSON (common with small local models) it falls back
to using the raw query — the pipeline never crashes on a parse failure.

### 2.2 RetrieverAgent (`agents/retriever_agent.py`)
Executes vector search for every rephrasing, deduplicates by chunk id, applies the
`SCORE_THRESHOLD`, keeps `TOP_K_FINAL`, then does an LLM relevance-filter pass.

### 2.3 AnalyzerAgent (`agents/analyzer_agent.py`)
Deep-reads retrieved chunks and extracts: key facts with source attribution,
supporting quotes, contradictions between sources, and **information gaps**
(what the query asks that the sources cannot answer). Output feeds synthesis.

### 2.4 SynthesizerAgent (`agents/synthesizer_agent.py`)
Writes the grounded answer. On a revision cycle it also receives the critic's
`REVISION_INSTRUCTIONS` and the original context, and produces an improved draft.

### 2.5 CriticAgent (`agents/critic_agent.py`)
Scores the draft 1–10 across accuracy / completeness / clarity / citations /
conciseness and emits `VERDICT: APPROVED | NEEDS_REVISION`. On NEEDS_REVISION the
graph loops back to synthesis, at most `MAX_REVISION_CYCLES` times — quality
control with a guaranteed termination bound.

### 2.6 Metrics
`utils/metrics.py` times every node (per revision iteration) and counts revision
cycles. Every result includes a `metrics` dict — that's your per-query latency
breakdown for monitoring.

---

## 3. The LLM Provider Layer (`llm/providers.py`)

Every agent calls exactly one function:

```python
from llm import chat
answer = chat(system_prompt, user_message, thinking=True)
```

`LLM_PROVIDER` selects the backend:

| Provider | What it is | Needs |
|---|---|---|
| `anthropic` | Claude via Anthropic API. `thinking=True` enables adaptive thinking. | `ANTHROPIC_API_KEY` |
| `ollama` | Local open-source models via [Ollama](https://ollama.com)'s OpenAI-compatible `/v1` endpoint | `ollama serve` + `ollama pull llama3.1:8b` |
| `lmstudio` | Local models via [LM Studio](https://lmstudio.ai)'s local server. If `LMSTUDIO_MODEL` is empty the first loaded model is auto-detected. | LM Studio running with server enabled |
| `openai` | **Any** OpenAI-compatible endpoint: vLLM, llama.cpp server, Groq, Together, OpenRouter, or OpenAI itself | `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` |

Design details:

- **Retries**: transient errors (connection, rate-limit, 5xx, timeout) retry up to
  `LLM_MAX_RETRIES` with exponential backoff (`LLM_RETRY_BACKOFF × 2^attempt`).
  Non-retryable errors (bad key, unknown model) fail immediately with an
  actionable message.
- **Fail-fast checks**: `check_provider()` (used by `/health`, `main.py --check`,
  `ingest_cli.py check`, `scripts/check.sh`) verifies reachability and that the
  configured model is actually available *before* you burn a pipeline run.
- **Singleton**: the client is built once per process (`get_provider()`).
- `thinking` is honored only by Anthropic; OpenAI-compatible providers ignore it.

Recommended local models (Ollama):

| Model | Size | Notes |
|---|---|---|
| `llama3.1:8b` | ~4.7 GB | Good default; follows the structured prompts well |
| `qwen2.5:14b` | ~9 GB | Better JSON discipline and reasoning if you have RAM |
| `mistral:7b` | ~4.1 GB | Fastest acceptable quality |
| `llama3.1:70b` | ~40 GB | Near-API quality, needs a big machine |

Small models occasionally break the critic's output format — the parsers in every
agent have deterministic fallbacks, so quality degrades gracefully instead of
crashing.

---

## 4. Data Plane

### 4.1 Sources of documents

1. **Hand-curated files** — drop `.txt`/`.md` into `multi_agent_rag/data/`.
2. **Open-source datasets** — `rag/dataset_loader.py` streams Hugging Face
   datasets and **materialises records as files** under
   `multi_agent_rag/corpus/<dataset>/<contenthash>.txt`.
3. **API uploads** — `POST /ingest/text` writes to `corpus/api_uploads/`.

Materialising datasets as files is deliberate:

- **Idempotent**: filenames are content hashes; re-running a loader writes only new records.
- **One ingestion path**: the indexer just syncs directories — same code for all sources.
- **Inspectable**: every indexed document is a plain file you can open.

Built-in dataset presets (`python ingest_cli.py list-datasets`):

| Preset | Source | Contents |
|---|---|---|
| `wikipedia-simple` | `wikimedia/wikipedia` (20231101.simple) | ~240k clean encyclopedic articles — best default |
| `wikipedia-en` | `wikimedia/wikipedia` (20231101.en) | Full English Wikipedia (6M+; bound with `--max-docs`) |
| `ag-news` | `ag_news` | 120k categorized news articles |
| `cc-news` | `cc_news` | 700k+ CommonCrawl news articles |
| `squad` | `squad` | Wikipedia reading-comprehension paragraphs (deduplicated) |
| `pubmed-summarization` | `ccdv/pubmed-summarization` | Biomedical research articles |

Any other HF dataset works generically:

```bash
./scripts/ingest.sh --hf-path cnn_dailymail --hf-config 3.0.0 \
                    --text-field article --max-docs 300
```

Datasets are **streamed** (`streaming=True`) — nothing is fully downloaded, so even
Wikipedia-scale sources respect `--max-docs` without filling your disk. Individual
bad records are skipped and counted, never fatal; the initial hub connection is
retried 3 times.

### 4.2 Chunking (`rag/document_store.py`)
Character-window chunking (`CHUNK_SIZE=600`, `CHUNK_OVERLAP=80`) with
sentence-boundary snapping: a chunk prefers to end at `". "`, `"\n\n"` etc. rather
than mid-sentence. Every chunk carries metadata (source path, chunk index, char
offsets) that flows all the way into retrieval results for citations.

### 4.3 Embeddings & vector backends (`rag/retriever.py`)
`all-MiniLM-L6-v2` (384-dim SentenceTransformer) by default — small, fast, strong
for its size. Swap via `EMBEDDING_MODEL` (then run `ingest_cli.py sync --force`).

| Backend | Index | Persistence | When |
|---|---|---|---|
| `faiss` (default) | IndexFlatIP — exact cosine | in-RAM (rebuilt fast from cache) | production, up to millions of chunks |
| `chroma` | HNSW — approximate | on disk `chroma_db/` | very large corpora, restart-heavy setups |
| `numpy` | brute force | none | prototyping, zero extra deps |

### 4.4 IndexManager — incremental indexing (`rag/index_manager.py`)

The heart of the *regular data injection & update* story. `sync()` does:

1. Scan `data/` + `corpus/` recursively for `.txt`/`.md` files.
2. Hash every file; diff against `index_state/manifest.json` → report of
   added / changed / removed files.
3. Rebuild the in-memory document store from disk (files are the single source
   of truth, so deletions and edits are handled for free).
4. Embed chunks **through the cache**: `index_state/embeddings.npz` maps
   `sha256(chunk text) → vector`. Only never-seen chunks hit the GPU/CPU encoder.
   Adding 10 documents to a 10,000-document corpus embeds ~10 documents' worth
   of chunks, not 10,010.
5. Build the vector index from the (mostly cached) embeddings and atomically
   persist manifest + cache (temp-file + rename — a crash can't corrupt them).

The cache key is the chunk *content* hash, so it survives file renames and is
automatically invalidated when you change `CHUNK_SIZE`/`CHUNK_OVERLAP` (different
chunk text ⇒ different hashes). Changing `EMBEDDING_MODEL` requires
`sync(force=True)` since the text doesn't change but the vectors must.

### 4.5 The regular update process

Continuous data injection is a **pull → sync → hot-swap** loop:

```
                (cron / scheduler / compose "data-updater" service)
                                    │
        1. ingest_cli.py refresh    │   re-pull every dataset previously ingested
           (writes only NEW files)  │   (provenance stored in corpus/*/_dataset.json)
                                    ▼
        2. IndexManager.sync()          embeds only new chunks (cache)
                                    │
                                    ▼
        3. POST /refresh (optional)     a running API builds a NEW index off to the
                                        side and atomically swaps it — zero downtime,
                                        in-flight queries keep the old index
```

Ways to run it (pick one):

- **Cron** (bare-metal/VM):
  ```
  0 * * * *  /path/to/repo/scripts/update_data.sh >> /path/to/repo/multi_agent_rag/logs/update.log 2>&1
  ```
  The script takes a lock file so runs never overlap, loads `.env`, and — if
  `RAG_API_URL` is set — notifies the running API to hot-swap its index.
- **Docker Compose scheduler**: `docker compose --profile scheduler up -d` runs the
  `data-updater` service on a `UPDATE_INTERVAL` (default hourly) loop.
- **On demand**: `make update`, or `POST /refresh` on the API.

---

## 5. REST API (`api.py`)

Interactive docs at `http://localhost:8000/docs` (Swagger UI).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness: LLM provider status + index readiness. `"ok"` / `"degraded"` |
| GET | `/stats` | Documents/chunks/backend/embedding-model stats |
| GET | `/datasets` | Available open-source dataset presets |
| POST | `/query` | `{"query": "..."}` → full pipeline result (answer, verdict, score, metrics) |
| POST | `/ingest/text` | `{"content": "...", "source": "..."}` → persist + background re-index |
| POST | `/ingest/dataset` | `{"name": "ag-news", "max_docs": 200}` → background ingest + re-index |
| POST | `/refresh` | `{"pull_datasets": true, "force": false}` → background dataset refresh + re-index |

Operational behaviour:

- On startup the app builds the index before accepting traffic (`lifespan`);
  `/query` returns **503** until ready.
- Re-indexing builds a *new* `IndexManager` and swaps it under a lock — queries
  never see a half-built index (blue/green at the index level).
- A second concurrent re-index is skipped, not queued (`_reindex_lock`).
- LLM failures surface as **502** with the provider error message; input
  validation (query length etc.) is Pydantic-enforced **422**.

Example session:

```bash
curl -s localhost:8000/health | jq .status
curl -s -X POST localhost:8000/query \
     -H 'Content-Type: application/json' \
     -d '{"query": "Compare FAISS and ChromaDB"}' | jq -r .final_answer
```

---

## 6. Configuration Reference

All settings live in `multi_agent_rag/config.py`; every one can be overridden by
env var or `.env` (repo root). See `.env.example` for the annotated template.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `ollama` \| `lmstudio` \| `openai` |
| `ANTHROPIC_API_KEY` / `CLAUDE_MODEL` | — / `claude-opus-4-8` | Anthropic settings |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434/v1` / `llama3.1:8b` | Ollama settings |
| `LMSTUDIO_BASE_URL` / `LMSTUDIO_MODEL` | `http://localhost:1234/v1` / auto | LM Studio settings |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI defaults | any OpenAI-compatible endpoint |
| `MAX_TOKENS` | `8192` | token budget per agent call |
| `LLM_TEMPERATURE` | `0.2` | OpenAI-compatible providers only |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` / `LLM_RETRY_BACKOFF` | `300` / `3` / `2.0` | resilience knobs |
| `MAX_REVISION_CYCLES` | `2` | critic → synthesizer loop bound |
| `VECTOR_BACKEND` | `faiss` | `faiss` \| `chroma` \| `numpy` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer id |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `600` / `80` | chunking (chars) |
| `TOP_K_RETRIEVE` / `TOP_K_FINAL` / `SCORE_THRESHOLD` | `8` / `5` / `0.25` | retrieval |
| `DATASET_NAME` / `DATASET_MAX_DOCS` / `DATASET_MIN_CHARS` | `wikipedia-simple` / `500` / `200` | dataset ingestion defaults |
| `DATA_DIR` / `CORPUS_DIR` / `INDEX_STATE_DIR` | package-relative | relocate data/state (e.g. to a mounted volume) |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | API server |
| `RAG_API_URL` | unset | lets `update_data.sh` hot-swap a running API |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` |

---

## 7. Scripts Reference (`scripts/`)

All scripts use `set -euo pipefail`, an ERR trap with line numbers, colored logging,
`.env` loading, and actionable failure messages. All are idempotent.

| Script | What it does |
|---|---|
| `setup.sh` | Finds Python ≥3.10, creates `.venv`, installs deps, creates `.env` from template, pre-downloads the embedding model, checks the provider. Re-runnable. |
| `run.sh [args…]` | Interactive CLI; passes args to `main.py` (`--query`, `--dataset`, `--backend`, `--check`, `--force-reindex`) |
| `serve.sh` | Starts uvicorn; checks the port is free first; `WORKERS=n` for multi-worker |
| `ingest.sh` | `list` shows presets; otherwise wraps `ingest_cli.py dataset` and re-indexes |
| `update_data.sh` | The cron target: lock-file protected dataset refresh + incremental sync + optional API hot-swap |
| `check.sh` | Probes Ollama (:11434) and LM Studio (:1234), then full provider + index health check |
| `lib.sh` | shared helpers (not executable directly) |

`ingest_cli.py` subcommands: `list-datasets`, `dataset`, `refresh`, `sync [--force]`, `check`.

---

## 8. Deployment

### 8.1 Bare metal / VM

```bash
./scripts/setup.sh
# edit .env → pick provider
./scripts/ingest.sh --name wikipedia-simple --max-docs 500
./scripts/serve.sh                      # or: WORKERS=2 ./scripts/serve.sh
crontab -e                              # add the update_data.sh line (see §4.5)
```

For process supervision use systemd:

```ini
# /etc/systemd/system/rag-api.service
[Unit]
Description=Multi-Agent RAG API
After=network.target

[Service]
WorkingDirectory=/opt/Multi_Agent_RAG
ExecStart=/opt/Multi_Agent_RAG/scripts/serve.sh
Restart=always
RestartSec=5
EnvironmentFile=/opt/Multi_Agent_RAG/.env

[Install]
WantedBy=multi-user.target
```

### 8.2 Docker

```bash
cp .env.example .env && $EDITOR .env

docker compose up -d --build                      # API only (cloud LLM)
docker compose --profile ollama up -d --build     # + local Ollama
docker compose exec ollama ollama pull llama3.1:8b
docker compose --profile scheduler up -d          # + hourly data updater
```

Image details: `python:3.12-slim`, layer-cached deps, embedding model pre-baked
(fast cold start), non-root user, container `HEALTHCHECK` against `/health`.
State (corpus, index cache, chroma, logs) lives in named volumes and survives
rebuilds. When the API runs in compose with the Ollama profile, it reaches Ollama
at `http://ollama:11434/v1` automatically (`OLLAMA_BASE_URL_DOCKER`); if Ollama
runs on the *host* instead, set `OLLAMA_BASE_URL_DOCKER=http://host.docker.internal:11434/v1`.

### 8.3 Scaling notes

- **Multiple uvicorn workers** each hold their own FAISS index (RAM × workers).
  For big corpora prefer `VECTOR_BACKEND=chroma` (shared on-disk index) or a
  dedicated vector DB.
- The pipeline is LLM-bound (~30–60 s/query with revision on API models; longer on
  small local models). Scale horizontally behind a load balancer; the app is
  stateless apart from the index, which every replica rebuilds from the shared
  corpus volume.
- Put an API gateway in front for auth/rate-limiting; the service itself does not
  authenticate requests.

---

## 9. Observability & Operations

- **Logs**: structured, to console + `multi_agent_rag/logs/rag_system.log`
  (`LOG_LEVEL=DEBUG` for prompt-level detail). Update-job output belongs in
  `logs/update.log` via the cron redirect.
- **Per-query metrics**: every response contains `metrics.steps` (seconds per
  agent per iteration), `revision_cycles`, `score`, `retrieved_count`. Ship these
  to your metrics store; alert on rising latency, falling critic scores, or
  `retrieved_count == 0` rates (retrieval failure signal).
- **Health**: `GET /health` (degraded vs ok), container HEALTHCHECK, `scripts/check.sh`.
- **Quality regression testing**: keep a fixed set of golden queries, run them
  through `POST /query` after data or model changes, and compare critic scores —
  see the [RAG evaluation guide](https://sabbirahmedsaqlain.github.io/articles/rag-performance-evaluation-guide/).

### Troubleshooting

| Symptom | Cause → Fix |
|---|---|
| `ANTHROPIC_API_KEY is not set` | Put the key in `.env`, or switch `LLM_PROVIDER=ollama` |
| `Cannot reach ollama at …` | `ollama serve` isn't running, or wrong URL in Docker (see §8.2) |
| `reports no loaded models` | `ollama pull llama3.1:8b` / load a model in LM Studio and enable the server |
| 503 from `/query` | Index still building at startup — poll `/health` |
| First run very slow | One-time downloads: torch wheel, embedding model, dataset stream |
| Changed `EMBEDDING_MODEL`, weird scores | Re-embed: `python ingest_cli.py sync --force` |
| Answers ignore new documents | Run `./scripts/update_data.sh` or `POST /refresh`; check `manifest.json` timestamps |
| Small local model gives low critic scores | Expected — try `qwen2.5:14b`, or lower `MAX_REVISION_CYCLES` to save time |
| Port 8000 busy | `API_PORT=8080 ./scripts/serve.sh` |

---

## 10. Extending the System

**New agent** — subclass `BaseAgent`, add a node + edges in `agents/graph.py`
(the README shows a `FactCheckerAgent` example). Because agents only use
`self._call`, new agents automatically work on all four LLM providers.

**New data source** — either materialise files into `corpus/<source>/` yourself
(hash-named, like `dataset_loader` does) and run `sync`, or add a preset to
`PRESETS` in `rag/dataset_loader.py`.

**New vector backend** — implement `_build_x`/`_search_x` in
`rag/retriever.py` following the FAISS pattern.

**New LLM provider** — anything OpenAI-compatible already works via
`LLM_PROVIDER=openai` + `OPENAI_BASE_URL`. A truly different API means a new
provider class in `llm/providers.py` with `chat`, `is_retryable`, `check`.
