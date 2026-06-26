import time
from dataclasses import dataclass, field
from .base_agent import BaseAgent
from .retriever_agent import RetrieverAgent
from .analyzer_agent import AnalyzerAgent
from .synthesizer_agent import SynthesizerAgent
from .critic_agent import CriticAgent
from rag import DocumentStore, VectorRetriever


@dataclass
class PipelineResult:
    query: str
    final_answer: str
    agent_outputs: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    success: bool = True
    error: str = ""


class OrchestratorAgent(BaseAgent):
    """Coordinates all specialized agents to answer a user query end-to-end."""

    name = "OrchestratorAgent"
    description = "Master coordinator that routes queries through the retrieval, analysis, synthesis, and critic pipeline."

    def __init__(self, document_store: DocumentStore):
        super().__init__()
        self.document_store = document_store
        self.retriever = VectorRetriever(document_store)
        self.retriever_agent = RetrieverAgent(self.retriever)
        self.analyzer_agent = AnalyzerAgent()
        self.synthesizer_agent = SynthesizerAgent()
        self.critic_agent = CriticAgent()
        self._index_built = False

    def build_index(self):
        print("[Orchestrator] Building vector index...")
        self.retriever.build_index()
        self._index_built = True
        stats = self.document_store.stats()
        print(f"[Orchestrator] Index ready — {stats['total_documents']} docs, {stats['total_chunks']} chunks")

    def run(self, query: str, verbose: bool = True) -> PipelineResult:
        if not self._index_built:
            self.build_index()

        start = time.time()

        def log(msg: str):
            if verbose:
                print(msg)

        try:
            log(f"\n{'='*60}")
            log(f"[Orchestrator] Processing query: {query}")
            log(f"{'='*60}")

            # Step 1: Retrieve
            log("\n[Step 1/4] RetrieverAgent — searching knowledge base...")
            retrieval_output = self.retriever_agent.run(query)
            log(f"  Found {len(retrieval_output['retrieved_chunks'])} chunks")

            # Step 2: Analyze
            log("\n[Step 2/4] AnalyzerAgent — analyzing retrieved content...")
            analysis_output = self.analyzer_agent.run(query, retrieval_output)

            # Step 3: Synthesize
            log("\n[Step 3/4] SynthesizerAgent — crafting response...")
            synthesis_output = self.synthesizer_agent.run(query, analysis_output)

            # Step 4: Critic review
            log("\n[Step 4/4] CriticAgent — reviewing for quality...")
            critic_output = self.critic_agent.run(
                query,
                synthesis_output,
                context=analysis_output.get("context_used", ""),
            )

            elapsed = time.time() - start
            log(f"\n[Orchestrator] Pipeline complete in {elapsed:.1f}s")

            return PipelineResult(
                query=query,
                final_answer=critic_output["final_answer"],
                agent_outputs={
                    "retrieval": retrieval_output,
                    "analysis": analysis_output,
                    "synthesis": synthesis_output,
                    "critic": critic_output,
                },
                elapsed_seconds=elapsed,
                success=True,
            )

        except Exception as e:
            elapsed = time.time() - start
            log(f"[Orchestrator] ERROR: {e}")
            return PipelineResult(
                query=query,
                final_answer="",
                elapsed_seconds=elapsed,
                success=False,
                error=str(e),
            )
