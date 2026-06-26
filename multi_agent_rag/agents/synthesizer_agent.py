from .base_agent import BaseAgent


class SynthesizerAgent(BaseAgent):
    """Synthesizes analysis into a coherent, comprehensive answer."""

    name = "SynthesizerAgent"
    description = "Combines insights from the analyzer to produce a well-structured, comprehensive answer."

    SYSTEM_PROMPT = """You are a synthesis expert. Your role is to take analytical findings from multiple sources
and craft a clear, comprehensive, and well-organized answer to the user's query.

Guidelines:
1. Start with a direct answer to the question
2. Provide supporting details and evidence from the sources
3. Organize information logically (use headers if needed)
4. Acknowledge uncertainty where it exists
5. Keep the answer focused and avoid unnecessary repetition
6. Cite sources when making specific claims

Produce a polished, reader-friendly response."""

    def run(self, query: str, analyzer_output: dict) -> dict:
        analysis = analyzer_output.get("analysis", "")
        context = analyzer_output.get("context_used", "")

        messages = [
            {
                "role": "user",
                "content": (
                    f"User Query: {query}\n\n"
                    f"Analysis from AnalyzerAgent:\n{analysis}\n\n"
                    f"Original Context:\n{context}\n\n"
                    "Now synthesize this into a clear, comprehensive answer for the user."
                ),
            }
        ]

        response = self._call_claude(self.SYSTEM_PROMPT, messages)

        return {
            "agent": self.name,
            "query": query,
            "synthesized_answer": response,
        }
