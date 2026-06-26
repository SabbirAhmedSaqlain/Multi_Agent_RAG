from .base_agent import BaseAgent

_SYSTEM = """You are an expert information analyst for a RAG system.

Given a user query and retrieved document chunks, produce a deep analytical report with these sections:

## Key Facts
Bullet list of the most important, directly relevant facts extracted from the sources.

## Supporting Evidence
Specific quotes and data points from the sources that support the key facts.
Always attribute: "According to [source]..."

## Contradictions or Uncertainties
Any conflicting information across sources, or claims that seem uncertain.

## Information Gaps
What the query asks for that is NOT answered by the retrieved context.

## Synthesis Notes
2-3 sentences summarising what a downstream writer needs to know to answer the query well.

Be precise and analytical. Extract only what is grounded in the sources."""


class AnalyzerAgent(BaseAgent):
    name = "AnalyzerAgent"

    def run(self, query: str, retrieval_output: dict) -> dict:
        chunks = retrieval_output.get("retrieved_chunks", [])
        if not chunks:
            return {
                "analysis": "Insufficient context to perform analysis.",
                "context_for_synthesis": "",
            }

        context = "\n\n".join(
            f"[Source: {c['metadata'].get('filename', 'unknown')}]\n{c['content']}"
            for c in chunks
        )
        user_msg = (
            f"User Query: {query}\n\n"
            f"Retrieval Analysis:\n{retrieval_output.get('retrieval_analysis', '')}\n\n"
            f"Document Chunks:\n{context}"
        )
        self.log.info("Analysing %d chunks for query", len(chunks))
        analysis = self._call(_SYSTEM, user_msg)
        return {
            "analysis": analysis,
            "context_for_synthesis": context,
        }
