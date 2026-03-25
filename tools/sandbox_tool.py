"""Sandbox tool — runs generated code inside the isolated Docker sandbox.

The sandbox container has no outbound network access and only has read-only
access to the output/ directory. This tool executes the generated RL code
in that sandbox and captures stdout/stderr.
"""

from __future__ import annotations

import subprocess


def run_in_sandbox(
    script_path: str = "output/rl_trading_agent.py",
    timeout: int = 300,
) -> dict:
    """Execute a Python script inside the rl-sandbox Docker container.

    Args:
        script_path: Path to the script (relative to /app inside the container).
        timeout: Maximum execution time in seconds.

    Returns:
        Dict with keys: success (bool), stdout (str), stderr (str), return_code (int).
    """
    cmd = [
        "docker", "exec", "rl-sandbox",
        "python", "-u", script_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Sandbox execution timed out after {timeout}s",
            "return_code": -1,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Docker not available — cannot run sandbox",
            "return_code": -1,
        }
