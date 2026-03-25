"""Entry point for the RL Trading Agent pipeline.

Loads configuration from environment, builds the LangGraph pipeline,
and runs the full research → code → verify → push loop.
"""

from __future__ import annotations

import os
import sys
import sqlite3
from datetime import datetime, timezone

from dotenv import load_dotenv

from graph import build_graph
from state import AgentState


# ── Database helpers ──────────────────────────────────────────────


DB_PATH = os.environ.get("DB_PATH", "/data/db/state.db")


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Initialise SQLite database, creating tables from schema if needed."""
    schema_path = os.path.join(os.path.dirname(__file__), "db", "schema.sql")
    conn = sqlite3.connect(db_path)
    if os.path.exists(schema_path):
        with open(schema_path) as f:
            conn.executescript(f.read())
    return conn


def log_run(conn: sqlite3.Connection, state: AgentState) -> None:
    """Persist the final pipeline state to the runs table."""
    conn.execute(
        """
        INSERT INTO runs (
            started_at, phase, proof_passed, retry_count,
            total_return, sharpe_ratio, max_drawdown, win_rate,
            github_pushed, github_commit_url, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            state.get("current_phase", "unknown"),
            state.get("proof_passed", False),
            state.get("retry_count", 0),
            state.get("total_return"),
            state.get("sharpe_ratio"),
            state.get("max_drawdown"),
            state.get("win_rate"),
            state.get("github_pushed", False),
            state.get("github_commit_url", ""),
            state.get("error", ""),
        ),
    )
    conn.commit()


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    """Run the full agent pipeline."""
    load_dotenv()

    # Validate required env vars
    required = ["TAVILY_API_KEY", "GITHUB_TOKEN", "GITHUB_USERNAME", "GITHUB_REPO"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"[main] ERROR: missing env vars: {', '.join(missing)}")
        print("[main] Copy .env.example to .env and fill in values")
        sys.exit(1)

    # Database
    conn = init_db()

    # Build and run the pipeline
    graph = build_graph()

    initial_state: AgentState = {
        "retry_count": 0,
        "max_retries": 3,
        "current_phase": "research",
        "proof_passed": False,
        "github_pushed": False,
    }

    print("[main] ══════════════════════════════════════════════")
    print("[main] RL Trading Agent — starting pipeline")
    print("[main] ══════════════════════════════════════════════")

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        final_state = {**initial_state, "error": str(e), "current_phase": "error"}
        print(f"[main] FATAL: {e}")

    # Log results
    log_run(conn, final_state)
    conn.close()

    # Summary
    print()
    print("[main] ══════════════════════════════════════════════")
    if final_state.get("error"):
        print(f"[main] Pipeline FAILED: {final_state['error']}")
    elif final_state.get("github_pushed"):
        print(f"[main] Pipeline COMPLETE — pushed to GitHub")
        print(f"[main] Commit: {final_state.get('github_commit_url', 'N/A')}")
    elif final_state.get("proof_passed"):
        print("[main] Proof passed but GitHub push skipped")
    else:
        retries = final_state.get("retry_count", 0)
        print(f"[main] Proof did not pass after {retries} retries")
    print("[main] ══════════════════════════════════════════════")


if __name__ == "__main__":
    main()
