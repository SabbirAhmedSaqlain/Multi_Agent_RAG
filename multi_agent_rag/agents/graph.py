"""
LangGraph-powered multi-agent RAG pipeline.

Graph topology:
  query_analysis
       │
       ▼
   retrieval
       │
       ▼
   analysis
       │
       ▼
  synthesis ◄──────────────┐
       │                   │ (NEEDS_REVISION + iterations < MAX)
       ▼                   │
    critic ────────────────┘
       │
   (APPROVED or max iterations)
       │
      END
"""
from __future__ import annotations
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from .query_analyzer import QueryAnalyzerAgent
from .retriever_agent import RetrieverAgent
from .analyzer_agent import AnalyzerAgent
from .synthesizer_agent import SynthesizerAgent
from .critic_agent import CriticAgent
from rag import VectorRetriever
from config import MAX_REVISION_CYCLES
from utils import get_logger, PipelineMetrics

log = get_logger("PipelineGraph")


# ── State ──────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    # Input
    query: str
    # After query analysis
    query_analysis: dict
    # After retrieval
    retrieved_chunks: list[dict]
    retrieval_analysis: str
    has_context: bool
    # After analysis
    deep_analysis: str
    context_for_synthesis: str
    # After synthesis
    draft_answer: str
    # After critic
    verdict: str
    score: int
    revision_instructions: str
    final_answer: str
    # Control
    iteration: int
    metrics: PipelineMetrics


# ── Node functions (one per agent) ────────────────────────────────────────

def node_query_analysis(state: RAGState, agents: dict) -> dict:
    metrics: PipelineMetrics = state["metrics"]
    metrics.start_step("query_analysis")
    result = agents["query_analyzer"].run(state["query"])
    metrics.end_step()
    return {"query_analysis": result}


def node_retrieval(state: RAGState, agents: dict) -> dict:
    metrics: PipelineMetrics = state["metrics"]
    metrics.start_step("retrieval")
    out = agents["retriever"].run(state["query_analysis"])
    metrics.end_step()
    return {
        "retrieved_chunks": out["retrieved_chunks"],
        "retrieval_analysis": out["retrieval_analysis"],
        "has_context": out["has_context"],
    }


def node_analysis(state: RAGState, agents: dict) -> dict:
    metrics: PipelineMetrics = state["metrics"]
    metrics.start_step("analysis")
    retrieval_output = {
        "retrieved_chunks": state["retrieved_chunks"],
        "retrieval_analysis": state["retrieval_analysis"],
    }
    out = agents["analyzer"].run(state["query"], retrieval_output)
    metrics.end_step()
    return {
        "deep_analysis": out["analysis"],
        "context_for_synthesis": out["context_for_synthesis"],
    }


def node_synthesis(state: RAGState, agents: dict) -> dict:
    metrics: PipelineMetrics = state["metrics"]
    step_name = f"synthesis_iter{state['iteration']}"
    metrics.start_step(step_name)
    analysis_output = {
        "analysis": state["deep_analysis"],
        "context_for_synthesis": state["context_for_synthesis"],
    }
    critique = state.get("revision_instructions", "")
    out = agents["synthesizer"].run(state["query"], analysis_output, critique=critique)
    metrics.end_step()
    return {"draft_answer": out["draft_answer"]}


def node_critic(state: RAGState, agents: dict) -> dict:
    metrics: PipelineMetrics = state["metrics"]
    step_name = f"critic_iter{state['iteration']}"
    metrics.start_step(step_name)
    out = agents["critic"].run(
        state["query"],
        state["draft_answer"],
        state["context_for_synthesis"],
    )
    metrics.end_step()
    iteration = state["iteration"] + 1
    if out["verdict"] == "NEEDS_REVISION":
        metrics.revision_count += 1
    return {
        "verdict": out["verdict"],
        "score": out["score"],
        "revision_instructions": out.get("revision_instructions", ""),
        "final_answer": out["final_answer"],
        "iteration": iteration,
    }


def route_after_critic(state: RAGState) -> str:
    if state["verdict"] == "NEEDS_REVISION" and state["iteration"] <= MAX_REVISION_CYCLES:
        log.info("Critic requested revision (iteration %d/%d)", state["iteration"], MAX_REVISION_CYCLES)
        return "revise"
    return "done"


# ── Graph builder ──────────────────────────────────────────────────────────

def build_graph(retriever: VectorRetriever) -> StateGraph:
    agents = {
        "query_analyzer": QueryAnalyzerAgent(),
        "retriever": RetrieverAgent(retriever),
        "analyzer": AnalyzerAgent(),
        "synthesizer": SynthesizerAgent(),
        "critic": CriticAgent(),
    }

    # Wrap node functions to inject agents
    def qa_node(state): return node_query_analysis(state, agents)
    def ret_node(state): return node_retrieval(state, agents)
    def ana_node(state): return node_analysis(state, agents)
    def syn_node(state): return node_synthesis(state, agents)
    def crit_node(state): return node_critic(state, agents)

    graph = StateGraph(RAGState)
    graph.add_node("query_analysis", qa_node)
    graph.add_node("retrieval", ret_node)
    graph.add_node("analysis", ana_node)
    graph.add_node("synthesis", syn_node)
    graph.add_node("critic", crit_node)

    graph.set_entry_point("query_analysis")
    graph.add_edge("query_analysis", "retrieval")
    graph.add_edge("retrieval", "analysis")
    graph.add_edge("analysis", "synthesis")
    graph.add_edge("synthesis", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"revise": "synthesis", "done": END},
    )

    return graph.compile()


# ── Top-level runner ───────────────────────────────────────────────────────

def run_pipeline(query: str, retriever: VectorRetriever) -> dict:
    log.info("=" * 60)
    log.info("Pipeline START — query: %s", query[:80])

    compiled = build_graph(retriever)
    metrics = PipelineMetrics()

    initial_state: RAGState = {
        "query": query,
        "query_analysis": {},
        "retrieved_chunks": [],
        "retrieval_analysis": "",
        "has_context": False,
        "deep_analysis": "",
        "context_for_synthesis": "",
        "draft_answer": "",
        "verdict": "",
        "score": 0,
        "revision_instructions": "",
        "final_answer": "",
        "iteration": 0,
        "metrics": metrics,
    }

    final_state = compiled.invoke(initial_state)
    report = metrics.report()
    log.info("Pipeline END — verdict=%s score=%d/10 time=%.1fs revisions=%d",
             final_state["verdict"], final_state["score"],
             report["total_seconds"], report["revision_cycles"])

    return {
        "query": query,
        "final_answer": final_state["final_answer"],
        "verdict": final_state["verdict"],
        "score": final_state["score"],
        "metrics": report,
        "retrieved_count": len(final_state["retrieved_chunks"]),
    }
