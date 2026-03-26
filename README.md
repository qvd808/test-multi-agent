# RL Trading Agent — Specification-Driven Autonomous Coding System

A multi-agent pipeline that autonomously researches, writes, and formally verifies a reinforcement learning paper trading agent. The system uses a **Specification-Driven** architecture: a human-provided Rocq (Coq) specification serves as the mathematical "Source of Truth" against which the AI-generated code is verified.

---

## Project Goals

- **Spec-Driven Generation**: AI implements RL agents based on formal mathematical models.
- **Formal Verification**: Prove structural invariants (bounds, non-negativity) in Rocq.
- **Differential Testing**: Verify the AI's Python code against a verified reference using fuzzy testing.
- **Autonomous Deployment**: Push verified code to GitHub without manual intervention.

---

## Verification Architecture

This project uses a **Correct-by-Construction** approach to bridge high-level math and executable code.

### 1. The Formal Specification (`proofs/trading_agent_proof.v`)
The developer (you) defines the environment's state transitions and invariants in Rocq. This file is the mathematical contract that the RL agent must follow.

### 2. The Verification Pipeline (`agents/verifier.py`)
Instead of just "running tests", the verifier performs:
1. **Spec Compilation**: Ensures the Rocq proof is mathematically sound via `coqc`.
2. **Logic Extraction**: Translates the formal Coq logic into a Python reference function.
3. **Differential Fuzzing**: Generates 100+ random market states and actions. It compares the AI agent's state transitions against the verified reference. If they deviate by even a fraction, verification fails and the agent is sent back to the coding phase with the exact failing trace.

---

## Architecture Overview

```
HOST MACHINE (native)
─────────────────────────────────────────────────
  Ollama + model (qwen2.5-coder:7b-instruct-q4_K_M)
  └── exposes port 11434 to Docker network

DOCKER COMPOSE NETWORK
─────────────────────────────────────────────────
  agent container        (Python, LangGraph, Rocq/opam)
        │         │              │
        │         │              └───────────► GitHub (Verified Code)
        │         │
        │         ▼
        │   SQLite container    (Logs, Metrics, Runs)
        │
        ▼
  RL sandbox container   (Backtesting, isolated)
```

---

## Technology Stack

| Component | Technology | Responsibility |
|---|---|---|
| Orchestration | LangGraph | State machine controlling agent flow |
| Formal Math | Rocq (Coq) | Defining verified environment logic |
| Verification | Differential Fuzzer | Validating AI code against the Math spec |
| LLM | Gemini 2.5 Flash | Thinking, Research, and Coding |
| Fallback LLM | Qwen 2.5 Coder (Ollama) | Local resilience |
| Persistence | SQLite | Storing run history and metrics |

---

## Development Phases

### Phase 1 — Specification
- [x] Define `AgentState` and `step` function in Rocq
- [x] Prove structural bounds (`shares_held_bounded`)
- [x] Establish the mathematical Source of Truth

### Phase 2 — Coding & Verification
- [x] Coder agent reads Rocq spec and implements `TradingEnv`
- [x] Verifier agent performs differential testing
- [x] Loop continues until Python logic matches Rocq logic 100%

### Phase 3 — Deployment
- [ ] Automated push to GitHub on verified pass
- [ ] Run backtests in the RL sandbox to measure Alpha

---

## License

MIT
