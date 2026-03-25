"""Coder agent — generates RL trading code from research context.

Takes the researcher's summary and produces a complete, runnable Python file
implementing a Gymnasium-based paper trading RL agent. On retries, it receives
feedback from the verifier about what went wrong with the Rocq proofs and
adjusts the code accordingly.
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama

from state import AgentState


SYSTEM_PROMPT = """You are an expert Python developer specialising in reinforcement learning.
Your job is to generate a complete, runnable RL paper trading agent using Gymnasium.

Requirements:
1. Use a standard RL algorithm (DQN with a simple neural network)
2. Implement a Gymnasium environment for paper trading with OHLCV data
3. Portfolio value must NEVER go below zero
4. Position sizes must stay within defined bounds (0 to max_position)
5. The reward function must always return a finite float value
6. State transitions must be deterministic given the same inputs
7. Include clear docstrings and type hints

Output ONLY valid Python code — no markdown fences, no explanations outside comments.
The code must be a single file that can be saved and run directly."""


RETRY_PROMPT_TEMPLATE = """The verifier found issues with your previous code.

Previous code:
```python
{previous_code}
```

Verifier feedback:
{feedback}

Fix the issues while maintaining all structural guarantees:
- Portfolio value never negative
- Position sizes within bounds
- Reward always finite
- Deterministic state transitions

Output ONLY the corrected Python code."""


def coder_node(state: AgentState) -> AgentState:
    """Generate or fix RL trading code."""
    retry_count = state.get("retry_count", 0)
    is_retry = bool(state.get("retry_feedback"))

    phase_label = f"retry {retry_count + 1}" if is_retry else "initial generation"
    print(f"[coder] Starting code generation ({phase_label})")

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")

    gemini_key = os.environ.get("GOOGLE_API_KEY")

    llm_ollama = ChatOllama(model=model_name, base_url=ollama_url, temperature=0.2)
    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, max_retries=0)
        llm = llm_gemini.with_fallbacks([llm_ollama])
    else:
        llm = llm_ollama

    if is_retry:
        user_prompt = RETRY_PROMPT_TEMPLATE.format(
            previous_code=state.get("generated_code", ""),
            feedback=state.get("retry_feedback", ""),
        )
    else:
        research = state.get("research_context", "(no research context available)")
        user_prompt = (
            f"Based on this research:\n\n{research}\n\n"
            "Generate a complete, single-file Python RL paper trading agent. "
            "The file should be runnable with `python <filename>` and include "
            "training + a simple backtest at the end."
        )

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    generated_code = response.content

    # Strip markdown code fences if the LLM included them
    if generated_code.startswith("```"):
        lines = generated_code.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        generated_code = "\n".join(lines)

    # Write to disk
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "rl_trading_agent.py")
    with open(filepath, "w") as f:
        f.write(generated_code)

    print(f"[coder] Wrote {len(generated_code)} chars to {filepath}")

    return {
        **state,
        "generated_code": generated_code,
        "code_filepath": filepath,
        "current_phase": "verify",
        "retry_count": retry_count + 1 if is_retry else 0,
    }
