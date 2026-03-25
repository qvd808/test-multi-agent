"""Shared state schema for the LangGraph agent pipeline.

All agents read from and write to this TypedDict. LangGraph passes the
full state between nodes, so every field must be serialisable.
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """State shared across the entire agent graph.

    Fields are grouped by the agent/phase that primarily writes them,
    but any node can read any field.
    """

    # ── Researcher ────────────────────────────────────────────────
    research_query: str          # The search query used
    research_context: str        # Aggregated research results (markdown)

    # ── Coder ─────────────────────────────────────────────────────
    generated_code: str          # Python source of the RL trading agent
    code_filepath: str           # Where the code was written on disk

    # ── Verifier ──────────────────────────────────────────────────
    proof_source: str            # Rocq (.v) source generated for verification
    proof_filepath: str          # Where the .v file was written
    proof_passed: bool           # True if coqc exited 0
    proof_error: str             # stderr from coqc on failure

    # ── Retry logic ───────────────────────────────────────────────
    retry_count: int             # Number of coder→verifier retries so far
    max_retries: int             # Upper bound (default 3)
    retry_feedback: str          # Verifier feedback sent back to coder

    # ── Metrics (paper trading) ───────────────────────────────────
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    cumulative_reward: float

    # ── GitHub ────────────────────────────────────────────────────
    github_pushed: bool          # True after successful push
    github_commit_url: str       # URL of the pushed commit

    # ── Pipeline control ──────────────────────────────────────────
    current_phase: str           # "research" | "code" | "verify" | "push" | "done"
    error: str                   # Fatal error message (stops the pipeline)
