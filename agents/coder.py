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

CRITICAL REQUIREMENT:
You MUST implement the environment logic (trading step) exactly as defined in the provided Coq mathematical specification. 
The verifier will run a differential fuzzer against your code and a verified reference extracted from Coq. 
If your logic deviates (e.g. different commission calculation, different position check), verification will fail.

Requirements:
1. Use a standard RL algorithm (DQN with a simple neural network)
2. Implement a Gymnasium environment for paper trading with OHLCV data
3. Follow the provided Coq specification for the `step` function logic
4. Include clear docstrings and type hints

Output ONLY valid Python code — no markdown fences, no explanations outside comments.
The code must be a single file that can be saved and run directly."""


RETRY_PROMPT_TEMPLATE = """The verifier found mismatches between your code and the verified specification.

Previous code:
```python
{previous_code}
```

Verifier feedback:
{feedback}

Fix the issues so that your `TradingEnv.step` method exactly matches the mathematical model in the Coq specification.

Output ONLY the corrected Python code."""


def coder_node(state: AgentState) -> AgentState:
    """Generate or fix RL trading code."""
    retry_count = state.get("retry_count", 0)
    is_retry = bool(state.get("retry_feedback"))

    phase_label = f"retry {retry_count + 1}" if is_retry else "initial generation"
    print(f"[coder] Starting code generation ({phase_label})")

    # Read the Coq specification to include in the prompt
    spec_path = os.path.join(os.path.dirname(__file__), "..", "proofs", "trading_agent_proof.v")
    try:
        with open(spec_path, "r") as f:
            coq_spec = f.read()
    except Exception:
        coq_spec = "No specification found."

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")

    gemini_key = os.environ.get("GOOGLE_API_KEY")

    llm_ollama = ChatOllama(model=model_name, base_url=ollama_url, temperature=0.2)
    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_retries=0)
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
            f"FORMAL SPECIFICATION (FOLLOW THIS EXACTLY):\n\n{coq_spec}\n\n"
            f"Based on this research:\n\n{research}\n\n"
            "Generate a complete, single-file Python RL paper trading agent. "
            "Implement the `TradingEnv.step` function to match the logic of the `step` function in the Coq spec above. "
            "The file should be runnable with `python <filename>` and include training + backtest."
        )

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    generated_code = response.content

    # Strip markdown code fences if the LLM included them
    if generated_code.strip().startswith("```"):
        lines = generated_code.strip().split("\n")
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
