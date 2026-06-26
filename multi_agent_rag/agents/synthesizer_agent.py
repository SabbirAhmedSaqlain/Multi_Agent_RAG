from .base_agent import BaseAgent

_SYSTEM = """You are an expert writer producing well-structured, accurate answers for a RAG system.

Rules:
1. Open with a direct, concise answer to the question
2. Structure with headers (##) if the answer is long
3. Every factual claim must be grounded in the provided analysis / context — never fabricate
4. Use "According to [source]..." attribution for specific data points
5. If information is incomplete, say so explicitly instead of guessing
6. End with a one-sentence summary

Style: authoritative but accessible. No bullet-list dumps — write in flowing prose unless a list genuinely helps."""

_REVISION_SYSTEM = """You are rewriting a draft answer based on critic feedback.

Apply ALL the critic's suggestions precisely. Keep what was marked as STRENGTHS.
Do not add new information not present in the original context.
Return only the improved answer, no preamble."""


class SynthesizerAgent(BaseAgent):
    name = "SynthesizerAgent"

    def run(self, query: str, analysis_output: dict, critique: str = "") -> dict:
        analysis = analysis_output.get("analysis", "")
        context = analysis_output.get("context_for_synthesis", "")

        if critique:
            self.log.info("Revising answer based on critic feedback")
            system = _REVISION_SYSTEM
            user_msg = (
                f"Original Query: {query}\n\n"
                f"Critic Feedback:\n{critique}\n\n"
                f"Analysis:\n{analysis}\n\n"
                f"Source Context:\n{context}"
            )
        else:
            self.log.info("Synthesising answer")
            system = _SYSTEM
            user_msg = (
                f"User Query: {query}\n\n"
                f"Analysis Report:\n{analysis}\n\n"
                f"Source Context:\n{context}"
            )

        answer = self._call(system, user_msg)
        return {"draft_answer": answer}
