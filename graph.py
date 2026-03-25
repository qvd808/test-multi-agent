"""LangGraph graph definition — wires agents together with conditional edges.

Graph flow:
    researcher → coder → verifier ─┬─ (proof passed) → push_to_github → END
                                    └─ (proof failed, retries left) → coder
                                    └─ (proof failed, no retries) → END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from state import AgentState
from agents.researcher import researcher_node
from agents.coder import coder_node
from agents.verifier import verifier_node
from tools.github_tool import push_to_github_node


# ── Routing logic ─────────────────────────────────────────────────


def after_verifier(state: AgentState) -> str:
    """Decide where to go after the verifier runs."""
    if state.get("error"):
        return END

    if state.get("proof_passed"):
        return "push_to_github"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    if retry_count < max_retries:
        return "coder"

    # Out of retries — stop
    return END


# ── Build the graph ───────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct and compile the agent pipeline graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("researcher", researcher_node)
    graph.add_node("coder", coder_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("push_to_github", push_to_github_node)

    # Linear edges
    graph.add_edge("researcher", "coder")
    graph.add_edge("coder", "verifier")
    graph.add_edge("push_to_github", END)

    # Conditional edge after verifier
    graph.add_conditional_edges("verifier", after_verifier)

    # Entry point
    graph.set_entry_point("researcher")

    return graph.compile()
