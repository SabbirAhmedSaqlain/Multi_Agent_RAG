from .base_agent import BaseAgent
from rag import VectorRetriever


class RetrieverAgent(BaseAgent):
    """Retrieves relevant documents from the knowledge base."""

    name = "RetrieverAgent"
    description = "Searches the document store and retrieves the most relevant chunks for a query."

    SYSTEM_PROMPT = """You are a retrieval specialist. Given a user query and a set of retrieved document chunks,
your job is to:
1. Analyze which retrieved chunks are most relevant to the query
2. Filter out irrelevant or low-quality chunks
3. Organize the relevant chunks clearly for downstream agents
4. Report any gaps if the retrieved information seems insufficient

Be concise and structured in your output. List the relevant excerpts and rate their relevance."""

    def __init__(self, retriever: VectorRetriever):
        super().__init__()
        self.retriever = retriever

    def run(self, query: str) -> dict:
        raw_results = self.retriever.retrieve(query)

        if not raw_results:
            return {
                "agent": self.name,
                "query": query,
                "retrieved_chunks": [],
                "analysis": "No documents found in the knowledge base.",
            }

        chunks_text = ""
        for i, r in enumerate(raw_results, 1):
            source = r["metadata"].get("source", "unknown")
            chunks_text += f"\n[Chunk {i}] (score={r['score']:.3f}, source={source})\n{r['content']}\n"

        messages = [
            {
                "role": "user",
                "content": f"Query: {query}\n\nRetrieved chunks:{chunks_text}\n\nAnalyze relevance and organize the useful information.",
            }
        ]

        analysis = self._call_claude(self.SYSTEM_PROMPT, messages)

        return {
            "agent": self.name,
            "query": query,
            "retrieved_chunks": raw_results,
            "analysis": analysis,
        }
