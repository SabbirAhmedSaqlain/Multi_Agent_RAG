from .base_agent import BaseAgent

_SYSTEM = """You are a rigorous answer quality critic for a RAG system.

Evaluate the draft answer against:
1. ACCURACY — Is every claim supported by the source context? Flag hallucinations.
2. COMPLETENESS — Does it address all aspects of the question?
3. CLARITY — Is it well-written, logically structured, and free of jargon?
4. CITATIONS — Are key facts properly attributed?
5. CONCISENESS — Is it appropriately sized (not too long, not too short)?

Output this exact structure:

VERDICT: APPROVED | NEEDS_REVISION
SCORE: <integer 1-10>

STRENGTHS:
- <bullet>

ISSUES:
- <bullet> (or "None")

REVISION_INSTRUCTIONS:
<Specific, actionable instructions for the synthesizer. Empty if APPROVED.>

FINAL_ANSWER:
<If APPROVED: copy the draft here verbatim.
 If NEEDS_REVISION: write the improved version yourself applying your own instructions.>"""


class CriticAgent(BaseAgent):
    name = "CriticAgent"

    def run(self, query: str, draft_answer: str, context: str) -> dict:
        self.log.info("Reviewing draft answer")
        user_msg = (
            f"Query: {query}\n\n"
            f"Source Context (ground truth):\n{context[:4000]}\n\n"
            f"Draft Answer to Review:\n{draft_answer}"
        )
        review = self._call(_SYSTEM, user_msg)

        verdict = "APPROVED" if "VERDICT: APPROVED" in review else "NEEDS_REVISION"
        score = self._extract_score(review)
        final = self._extract_section(review, "FINAL_ANSWER:", fallback=draft_answer)
        instructions = self._extract_section(review, "REVISION_INSTRUCTIONS:", fallback="")

        self.log.info("Critic verdict=%s score=%d/10", verdict, score)
        return {
            "verdict": verdict,
            "score": score,
            "review": review,
            "revision_instructions": instructions,
            "final_answer": final,
        }

    @staticmethod
    def _extract_score(text: str) -> int:
        import re
        m = re.search(r"SCORE:\s*(\d+)", text)
        return int(m.group(1)) if m else 7

    @staticmethod
    def _extract_section(text: str, marker: str, fallback: str) -> str:
        if marker in text:
            after = text.split(marker, 1)[1]
            # If there is a next section marker, stop there
            import re
            next_section = re.search(r"\n[A-Z_]+:\s*\n", after)
            if next_section:
                return after[:next_section.start()].strip()
            return after.strip()
        return fallback
