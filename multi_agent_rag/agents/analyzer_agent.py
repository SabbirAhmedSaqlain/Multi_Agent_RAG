from .base_agent import BaseAgent


class AnalyzerAgent(BaseAgent):
    """Analyzes retrieved documents to extract key facts and insights."""

    name = "AnalyzerAgent"
    description = "Deeply analyzes retrieved chunks to extract key facts, entities, and insights relevant to the query."

    SYSTEM_PROMPT = """You are an expert information analyst. Given a user query and retrieved document excerpts,
your task is to:
1. Identify the key facts, concepts, and entities directly relevant to the query
2. Note any contradictions or conflicting information across sources
3. Highlight the most important insights
4. Flag any information gaps that would be needed to fully answer the query
5. Summarize what is known and what is uncertain

Be thorough and analytical. Use bullet points and structured output."""

    def run(self, query: str, retrieved_data: dict) -> dict:
        chunks = retrieved_data.get("retrieved_chunks", [])
        retriever_analysis = retrieved_data.get("analysis", "")

        if not chunks:
            return {
                "agent": self.name,
                "query": query,
                "key_facts": [],
                "analysis": "No information available to analyze.",
                "gaps": ["No documents found in knowledge base"],
            }

        context = "\n\n".join(
            f"[Source {i+1}]: {c['content']}" for i, c in enumerate(chunks)
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"User Query: {query}\n\n"
                    f"Retriever Analysis:\n{retriever_analysis}\n\n"
                    f"Raw Document Chunks:\n{context}\n\n"
                    "Perform a deep analysis of this information to extract key facts, insights, and gaps."
                ),
            }
        ]

        analysis = self._call_claude(self.SYSTEM_PROMPT, messages)

        return {
            "agent": self.name,
            "query": query,
            "context_used": context,
            "analysis": analysis,
        }
