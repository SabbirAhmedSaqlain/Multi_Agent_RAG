from __future__ import annotations
from .base_agent import BaseAgent
from rag import VectorRetriever
from config import TOP_K_RETRIEVE

_SYSTEM = """You are a retrieval quality specialist.
Given a user query and retrieved document excerpts, your task is to:
1. Filter out excerpts that are not relevant to the query
2. Rank the remaining excerpts from most to least relevant
3. For each kept excerpt, state WHY it is relevant in one sentence
4. Note any information gaps (what the query asks for that is NOT covered)

Output format (markdown):
## Relevant Excerpts
[Source: <source>] (Relevance: HIGH/MEDIUM)
> <brief quote or paraphrase>
Reason: <one sentence>

## Information Gaps
- <gap 1>
- <gap 2> (or "None" if all aspects are covered)"""


class RetrieverAgent(BaseAgent):
    name = "RetrieverAgent"

    def __init__(self, retriever: VectorRetriever):
        super().__init__()
        self.retriever = retriever

    def run(self, query_analysis: dict) -> dict:
        original = query_analysis["original_query"]
        search_queries = query_analysis.get("search_queries", [original])

        # Retrieve for each rephrased query and deduplicate by chunk_id
        seen: set[str] = set()
        all_chunks: list[dict] = []
        for q in search_queries:
            self.log.debug("Retrieving for: %s", q[:60])
            for chunk in self.retriever.retrieve(q, top_k=TOP_K_RETRIEVE):
                if chunk["chunk_id"] not in seen:
                    seen.add(chunk["chunk_id"])
                    all_chunks.append(chunk)

        # Sort by score descending
        all_chunks.sort(key=lambda x: x["score"], reverse=True)
        self.log.info("Retrieved %d unique chunks for query", len(all_chunks))

        if not all_chunks:
            return {
                "retrieved_chunks": [],
                "retrieval_analysis": "No relevant documents found in the knowledge base.",
                "has_context": False,
            }

        # Build context string for the analysis LLM call
        context_str = ""
        for i, c in enumerate(all_chunks, 1):
            source = c["metadata"].get("filename", c["metadata"].get("source", "unknown"))
            context_str += f"\n[{i}] (score={c['score']:.3f}, source={source})\n{c['content']}\n"

        user_msg = f"Query: {original}\n\nRetrieved excerpts:{context_str}"
        analysis = self._call(_SYSTEM, user_msg, thinking=False)

        return {
            "retrieved_chunks": all_chunks,
            "retrieval_analysis": analysis,
            "has_context": True,
        }
