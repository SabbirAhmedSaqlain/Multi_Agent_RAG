from .base_agent import BaseAgent

_SYSTEM = """You are a query analysis expert for a RAG (Retrieval Augmented Generation) system.

Your job is to analyse the user's query and output a JSON object with these keys:
{
  "intent": "<one of: factual | analytical | comparative | exploratory | definition>",
  "search_queries": ["<rephrased query 1>", "<rephrased query 2>", "<rephrased query 3>"],
  "key_concepts": ["concept1", "concept2", ...],
  "requires_detail": <true | false>,
  "notes": "<any special handling needed>"
}

search_queries: 2-3 diverse rephrasings optimised for vector search (different angles on the same question).
key_concepts: the most important entities/topics for retrieval filtering.
requires_detail: true if a comprehensive answer is expected; false for a short factual answer.

Return ONLY the JSON object, no other text."""


class QueryAnalyzerAgent(BaseAgent):
    name = "QueryAnalyzer"

    def run(self, query: str) -> dict:
        self.log.info("Analysing query: %s", query[:80])
        raw = self._call(_SYSTEM, f"Query: {query}", thinking=False)
        import json, re
        try:
            # Strip markdown code blocks if present
            raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
            raw = re.sub(r"```$", "", raw.strip())
            result = json.loads(raw)
        except json.JSONDecodeError:
            self.log.warning("JSON parse failed, using defaults")
            result = {
                "intent": "factual",
                "search_queries": [query],
                "key_concepts": [],
                "requires_detail": True,
                "notes": "",
            }
        result["original_query"] = query
        self.log.info("Intent=%s | queries=%d | concepts=%s",
                      result.get("intent"), len(result.get("search_queries", [])),
                      result.get("key_concepts"))
        return result
