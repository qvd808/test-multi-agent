"""GitHub tool — push verified code to a GitHub repository via PyGithub.

Used as a LangGraph node: after the verifier passes, this pushes the
generated RL code and proof files to the configured GitHub repo.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from github import Github, GithubException

from state import AgentState


def push_to_github_node(state: AgentState) -> AgentState:
    """Push generated code and proofs to GitHub."""
    print("[github] Starting GitHub push")

    token = os.environ.get("GITHUB_TOKEN", "")
    username = os.environ.get("GITHUB_USERNAME", "")
    repo_name = os.environ.get("GITHUB_REPO", "rl-trading-agent-output")

    if not token or not username:
        return {
            **state,
            "github_pushed": False,
            "error": "GITHUB_TOKEN or GITHUB_USERNAME not set",
        }

    try:
        g = Github(token)
        repo = g.get_repo(f"{username}/{repo_name}")
    except GithubException as e:
        return {
            **state,
            "github_pushed": False,
            "error": f"Failed to access repo {username}/{repo_name}: {e}",
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch = "main"

    files_to_push: list[tuple[str, str]] = []

    # Generated RL code
    code = state.get("generated_code", "")
    if code:
        files_to_push.append((f"agents/{timestamp}/rl_trading_agent.py", code))

    # Rocq proof
    proof = state.get("proof_source", "")
    if proof:
        files_to_push.append((f"proofs/{timestamp}/trading_agent_proof.v", proof))

    if not files_to_push:
        return {
            **state,
            "github_pushed": False,
            "error": "No files to push",
        }

    commit_message = (
        f"Verified RL trading agent — {timestamp}\n\n"
        f"Proof passed: {state.get('proof_passed', False)}\n"
        f"Retries: {state.get('retry_count', 0)}\n"
        f"Sharpe ratio: {state.get('sharpe_ratio', 'N/A')}"
    )

    try:
        commit_url = ""
        for filepath, content in files_to_push:
            try:
                # Try to get existing file (update)
                existing = repo.get_contents(filepath, ref=branch)
                result = repo.update_file(
                    filepath, commit_message, content, existing.sha, branch=branch
                )
            except GithubException:
                # File doesn't exist yet (create)
                result = repo.create_file(
                    filepath, commit_message, content, branch=branch
                )
            commit_url = result["commit"].html_url

        print(f"[github] ✓ Pushed {len(files_to_push)} files to {username}/{repo_name}")
        print(f"[github] Commit: {commit_url}")

        return {
            **state,
            "github_pushed": True,
            "github_commit_url": commit_url,
            "current_phase": "done",
        }

    except GithubException as e:
        return {
            **state,
            "github_pushed": False,
            "error": f"GitHub push failed: {e}",
        }
