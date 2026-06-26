from .base_agent import BaseAgent


class CriticAgent(BaseAgent):
    """Reviews and validates the synthesized answer for quality and accuracy."""

    name = "CriticAgent"
    description = "Critically evaluates the synthesized answer for accuracy, completeness, and quality."

    SYSTEM_PROMPT = """You are a rigorous quality control expert and critic. Review the synthesized answer against:
1. The original user query — does it fully address what was asked?
2. The source documents — are all claims supported by evidence?
3. Completeness — are there important gaps or missing points?
4. Clarity — is the answer well-written and easy to understand?
5. Accuracy — are there any hallucinations or unsupported claims?

Output your review in this structured format:
- VERDICT: APPROVED / NEEDS_REVISION
- SCORE: X/10
- STRENGTHS: (bullet list)
- ISSUES: (bullet list, empty if none)
- SUGGESTIONS: (specific improvements if NEEDS_REVISION)
- FINAL_ANSWER: (the improved or approved answer text)"""

    def run(self, query: str, synthesized_output: dict, context: str) -> dict:
        synthesized_answer = synthesized_output.get("synthesized_answer", "")

        messages = [
            {
                "role": "user",
                "content": (
                    f"Original Query: {query}\n\n"
                    f"Source Context:\n{context}\n\n"
                    f"Synthesized Answer to Review:\n{synthesized_answer}\n\n"
                    "Critically evaluate this answer and provide your structured review."
                ),
            }
        ]

        review = self._call_claude(self.SYSTEM_PROMPT, messages)

        # Extract the final answer from the critic's output
        final_answer = self._extract_final_answer(review, synthesized_answer)

        return {
            "agent": self.name,
            "query": query,
            "review": review,
            "final_answer": final_answer,
        }

    def _extract_final_answer(self, review: str, fallback: str) -> str:
        marker = "FINAL_ANSWER:"
        if marker in review:
            return review.split(marker, 1)[1].strip()
        return fallback
