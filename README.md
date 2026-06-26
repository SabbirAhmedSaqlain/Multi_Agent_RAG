# Multi-Agent RAG System

A production-quality **Retrieval Augmented Generation (RAG)** system built with multiple specialized AI agents that collaborate to answer questions from your document knowledge base. Each agent has a distinct role and the Orchestrator coordinates them into a seamless pipeline.

---

## Architecture Overview

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                     OrchestratorAgent                        │
│  (Coordinates the full pipeline, manages agent lifecycle)    │
└──────────┬───────────────────────────────────────────────────┘
           │
           │  Step 1
           ▼
┌──────────────────────┐      ┌──────────────────────────────┐
│   RetrieverAgent     │◄────►│   Vector Store (FAISS-style) │
│                      │      │   sentence-transformers      │
│  • Vector search     │      │   cosine similarity          │
│  • Relevance filter  │      └──────────────────────────────┘
│  • Source ranking    │
└──────────┬───────────┘
           │  Step 2
           ▼
┌───────────────────────┐
│   AnalyzerAgent       │
│                       │
│  • Key fact extract   │
│  • Entity linking     │
│  • Gap detection      │
│  • Contradiction check│
└──────────┬────────────┘
           │  Step 3
           ▼
┌──────────────────────┐
│  SynthesizerAgent    │
│                      │
│  • Answer drafting   │
│  • Source citation   │
│  • Logical structure │
└──────────┬───────────┘
           │  Step 4
           ▼
┌──────────────────────┐
│    CriticAgent       │
│                      │
│  • Quality scoring   │
│  • Hallucin. check   │
│  • Final revision    │
└──────────┬───────────┘
           │
           ▼
      Final Answer
```

### Agents

| Agent | Role | Key Responsibilities |
|---|---|---|
| **OrchestratorAgent** | Pipeline coordinator | Initializes all agents, routes data between them, manages the end-to-end flow |
| **RetrieverAgent** | Document search | Performs vector similarity search, filters relevant chunks, ranks by relevance |
| **AnalyzerAgent** | Deep analysis | Extracts key facts, identifies entities, detects contradictions and gaps |
| **SynthesizerAgent** | Answer composer | Produces a coherent, well-structured answer grounded in the evidence |
| **CriticAgent** | Quality control | Validates accuracy, scores quality, revises if needed, outputs final answer |

---

## RAG Pipeline

### 1. Document Ingestion
Documents are split into overlapping chunks (configurable size and overlap). This ensures context isn't lost at chunk boundaries.

```
Document → Chunker → [chunk_0, chunk_1, chunk_2, ...]
                            │
                            ▼
                     SentenceTransformer
                     (all-MiniLM-L6-v2)
                            │
                            ▼
                     Embedding Vectors
                     (stored in memory)
```

### 2. Retrieval
At query time, the query is embedded and cosine similarity is computed against all stored chunk embeddings. The top-K most similar chunks are returned.

### 3. Multi-Agent Processing
Retrieved chunks pass through four agents sequentially, each adding value before passing results forward.

---

## Project Structure

```
MultiAgent/
├── README.md
└── multi_agent_rag/
    ├── main.py                  # Entry point with sample queries
    ├── config.py                # Configuration (model, chunk size, etc.)
    ├── requirements.txt
    ├── agents/
    │   ├── __init__.py
    │   ├── base_agent.py        # Base class (Claude client, streaming)
    │   ├── orchestrator.py      # OrchestratorAgent — pipeline coordinator
    │   ├── retriever_agent.py   # RetrieverAgent — document search
    │   ├── analyzer_agent.py    # AnalyzerAgent — fact extraction
    │   ├── synthesizer_agent.py # SynthesizerAgent — answer composition
    │   └── critic_agent.py      # CriticAgent — quality validation
    ├── rag/
    │   ├── __init__.py
    │   ├── document_store.py    # Document loading, chunking, storage
    │   └── retriever.py         # Vector embeddings + cosine similarity search
    └── data/                    # Place your documents here
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- An Anthropic API key ([get one here](https://console.anthropic.com))

### Install

```bash
# Clone / navigate to the project
cd MultiAgent

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r multi_agent_rag/requirements.txt
```

### Configure API Key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or add it to a `.env` file and load it:

```bash
echo 'ANTHROPIC_API_KEY=your-key-here' > .env
source .env
```

---

## Usage

### Run the Demo

```bash
cd multi_agent_rag
python main.py
```

The demo loads four built-in documents (climate science, renewable energy, machine learning, space exploration) and runs three example queries through the full multi-agent pipeline.

### Use Your Own Documents

```python
from rag import DocumentStore
from agents import OrchestratorAgent

store = DocumentStore()

# Add documents from text strings
store.add_document("Your document text here...", source="my_doc.txt")

# Add from a file
store.add_from_file("/path/to/document.txt")

# Load all .txt and .md files from a directory
store.load_from_directory("./data", extensions=[".txt", ".md"])

# Initialize and query
orchestrator = OrchestratorAgent(store)
result = orchestrator.run("What does the document say about X?")

print(result.final_answer)
```

### Access Intermediate Agent Outputs

```python
result = orchestrator.run("Your query here")

# Retriever output
print(result.agent_outputs["retrieval"]["analysis"])

# Analyzer output
print(result.agent_outputs["analysis"]["analysis"])

# Synthesizer draft
print(result.agent_outputs["synthesis"]["synthesized_answer"])

# Critic review with score
print(result.agent_outputs["critic"]["review"])

# Final validated answer
print(result.final_answer)

# Timing
print(f"Completed in {result.elapsed_seconds:.1f}s")
```

### Silent Mode (no console logs)

```python
result = orchestrator.run("Your query", verbose=False)
```

---

## Configuration

Edit `multi_agent_rag/config.py`:

```python
CLAUDE_MODEL = "claude-opus-4-8"      # Anthropic model to use
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
MAX_TOKENS = 8192                      # Max tokens per agent call
CHUNK_SIZE = 500                       # Characters per document chunk
CHUNK_OVERLAP = 50                     # Overlap between consecutive chunks
TOP_K_RESULTS = 5                      # Number of chunks to retrieve per query
```

### Model Options

| Model | Use Case | Cost |
|---|---|---|
| `claude-opus-4-8` | Best quality, complex reasoning | Higher |
| `claude-sonnet-4-6` | Balanced quality/speed | Medium |
| `claude-haiku-4-5` | Fast, simple queries | Lower |

---

## How the Agents Collaborate

### Data Flow

```
query
  │
  ├──► RetrieverAgent.run(query)
  │         └── {retrieved_chunks, analysis}
  │
  ├──► AnalyzerAgent.run(query, retrieval_output)
  │         └── {context_used, analysis}
  │
  ├──► SynthesizerAgent.run(query, analysis_output)
  │         └── {synthesized_answer}
  │
  └──► CriticAgent.run(query, synthesis_output, context)
            └── {review, final_answer}
```

### Agent System Prompts

Each agent is given a specialized system prompt that shapes its behavior:

- **RetrieverAgent** — focuses on relevance scoring and filtering noise
- **AnalyzerAgent** — focuses on systematic fact extraction and gap detection
- **SynthesizerAgent** — focuses on clear, citation-backed answer composition
- **CriticAgent** — focuses on hallucination detection and quality scoring (outputs VERDICT: APPROVED / NEEDS_REVISION with a 1-10 score)

### Thinking Mode

All agents use `thinking: {type: "adaptive"}` with Claude Opus 4.8. This enables the model to silently reason before responding, significantly improving quality for multi-step analytical tasks.

---

## Example Output

```
============================================================
[Orchestrator] Processing query: How does RAG work and why is it useful?
============================================================

[Step 1/4] RetrieverAgent — searching knowledge base...
  Found 5 chunks

[Step 2/4] AnalyzerAgent — analyzing retrieved content...

[Step 3/4] SynthesizerAgent — crafting response...

[Step 4/4] CriticAgent — reviewing for quality...

[Orchestrator] Pipeline complete in 18.3s

============================================================
FINAL ANSWER:
============================================================
Retrieval Augmented Generation (RAG) is a technique that enhances Large Language
Models by connecting them to external knowledge bases at inference time.

**How RAG Works:**
1. A user query is embedded into a vector representation
2. The system searches a document store for semantically similar chunks
3. The most relevant chunks are retrieved and injected into the LLM prompt
4. The LLM generates an answer grounded in the retrieved evidence

**Why RAG is Useful:**
- Reduces hallucinations by grounding answers in real documents
- Allows LLMs to access up-to-date information beyond their training cutoff
- Enables domain-specific Q&A without expensive model fine-tuning
- Provides source citations for verifiable answers

[Completed in 18.3s]
```

---

## Extending the System

### Adding a New Agent

1. Create `agents/my_agent.py` inheriting from `BaseAgent`
2. Implement the `run()` method
3. Register it in `agents/__init__.py`
4. Add a step in `orchestrator.py`

```python
from .base_agent import BaseAgent

class MyAgent(BaseAgent):
    name = "MyAgent"
    description = "Does something specialized."

    SYSTEM_PROMPT = "You are a specialist in..."

    def run(self, query: str, prior_output: dict) -> dict:
        messages = [{"role": "user", "content": f"Query: {query}\n\n..."}]
        response = self._call_claude(self.SYSTEM_PROMPT, messages)
        return {"agent": self.name, "output": response}
```

### Swap the Embedding Model

Change `EMBEDDING_MODEL` in `config.py` to any [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) model:

```python
EMBEDDING_MODEL = "all-mpnet-base-v2"   # Higher quality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # Fast (default)
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"  # Optimized for Q&A
```

### Persistent Vector Store

For production use, replace the in-memory numpy store with [ChromaDB](https://www.trychroma.com/) or [Qdrant](https://qdrant.tech/) in `rag/retriever.py`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Official Anthropic Python SDK for Claude API |
| `sentence-transformers` | Text embedding models |
| `numpy` | Vector math and cosine similarity |
| `torch` | Required by sentence-transformers |

---

## License

MIT License — free to use, modify, and distribute.
