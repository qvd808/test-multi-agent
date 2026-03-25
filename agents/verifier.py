"""Verifier agent — generates and checks Rocq proofs for the generated code.

The verifier asks the LLM to produce a Rocq (.v) proof stub that encodes
structural properties of the generated code:
  - Portfolio value ≥ 0
  - Position sizes within bounds
  - Reward returns finite value
  - Deterministic state transitions

It then runs `coqc` to check the proof. If the proof compiles, verification
passes. If not, the error is fed back to the coder for retry.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from langchain_ollama import ChatOllama

from state import AgentState


SYSTEM_PROMPT = """You are a formal verification expert using the Rocq proof assistant (Coq).
Your job is to write Rocq proof scripts that verify structural properties of a Python
reinforcement learning trading agent.

You will be given Python source code and must produce a .v file that:
1. Defines abstract types matching the Python code's structures
2. States and proves key invariants:
   - portfolio_value_non_negative: portfolio value is always ≥ 0
   - position_within_bounds: position sizes stay within [0, max_position]
   - reward_is_finite: reward function returns a finite real number
   - deterministic_step: same state + action → same next_state

Output ONLY valid Rocq/Coq source code. No markdown, no explanations.
Start with appropriate Require Import statements.
IMPORTANT: The environment runs Coq 8.12.2. DO NOT use `Require Import Coq.Init.Prim.`, as it does not exist in this version.
Use standard libraries like `Coq.Reals.Reals`, `Coq.Init.Nat`, etc."""


def verifier_node(state: AgentState) -> AgentState:
    """Generate a Rocq proof and attempt to compile it."""
    print("[verifier] Starting verification phase")

    generated_code = state.get("generated_code", "")
    if not generated_code:
        return {
            **state,
            "proof_passed": False,
            "proof_error": "No generated code to verify",
            "retry_feedback": "No code was generated. Please generate the trading agent code.",
        }

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")

    gemini_key = os.environ.get("GOOGLE_API_KEY")

    llm_ollama = ChatOllama(model=model_name, base_url=ollama_url, temperature=0.1)
    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1, max_retries=0)
        llm = llm_gemini.with_fallbacks([llm_ollama])
    else:
        llm = llm_ollama

    prompt = (
        f"Here is a Python RL trading agent:\n\n```python\n{generated_code}\n```\n\n"
        "Write a Rocq (Coq) proof script that verifies the structural properties:\n"
        "1. Portfolio value never negative\n"
        "2. Position sizes within bounds\n"
        "3. Reward function returns finite value\n"
        "4. State transitions are deterministic\n\n"
        "Use Coq.Reals and basic Coq libraries. Keep proofs simple and constructive."
    )

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    proof_source = response.content

    # Strip markdown fences if present
    if proof_source.startswith("```"):
        lines = proof_source.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        proof_source = "\n".join(lines)

    # Write proof to disk
    proofs_dir = os.path.join(os.path.dirname(__file__), "..", "proofs")
    os.makedirs(proofs_dir, exist_ok=True)
    proof_path = os.path.join(proofs_dir, "trading_agent_proof.v")
    with open(proof_path, "w") as f:
        f.write(proof_source)

    print(f"[verifier] Wrote proof to {proof_path}")

    # Attempt to compile with coqc
    try:
        result = subprocess.run(
            ["coqc", proof_path],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print("[verifier] ✓ Proof compiled successfully")
            return {
                **state,
                "proof_source": proof_source,
                "proof_filepath": proof_path,
                "proof_passed": True,
                "proof_error": "",
                "retry_feedback": "",
                "current_phase": "push",
            }
        else:
            error_msg = result.stderr or result.stdout
            print(f"[verifier] ✗ Proof failed:\n{error_msg[:500]}")
            return {
                **state,
                "proof_source": proof_source,
                "proof_filepath": proof_path,
                "proof_passed": False,
                "proof_error": error_msg,
                "retry_feedback": (
                    f"Rocq proof compilation failed. Error:\n{error_msg}\n\n"
                    "Please adjust the Python code so that these structural properties "
                    "can be more easily verified. Ensure explicit bounds checking, "
                    "non-negative portfolio enforcement, and deterministic transitions."
                ),
                "current_phase": "code",
            }

    except FileNotFoundError:
        msg = "coqc not found — Rocq/Coq is not installed in this environment"
        print(f"[verifier] ✗ {msg}")
        return {
            **state,
            "proof_source": proof_source,
            "proof_filepath": proof_path,
            "proof_passed": False,
            "proof_error": msg,
            "retry_feedback": msg,
        }
    except subprocess.TimeoutExpired:
        msg = "coqc timed out after 120 seconds"
        print(f"[verifier] ✗ {msg}")
        return {
            **state,
            "proof_source": proof_source,
            "proof_filepath": proof_path,
            "proof_passed": False,
            "proof_error": msg,
            "retry_feedback": f"{msg}. Simplify the proof obligations.",
        }
