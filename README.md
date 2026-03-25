# RL Trading Agent — Multi-Agent Autonomous Coding System

A multi-agent pipeline that autonomously researches, writes, and formally verifies a reinforcement learning paper trading agent. The system uses LangGraph for orchestration, local LLMs via Ollama for inference, Rocq for formal verification of code properties, and a paper trading backtesting layer for empirical performance measurement.

---

## Project Goals

- Autonomously generate a reinforcement learning agent that trades paper stocks
- Formally verify structural properties of the generated code using Rocq
- Measure trading performance empirically via backtesting metrics
- Push verified code to GitHub without manual intervention
- Run entirely on local hardware with no paid API dependencies (except Tavily free tier)

---

## Architecture Overview

```
HOST MACHINE (native)
─────────────────────────────────────────────────
  Ollama + model (qwen2.5-coder:7b-instruct-q4_K_M)
  └── exposes port 11434 to Docker network
      via host.docker.internal

DOCKER COMPOSE NETWORK
─────────────────────────────────────────────────
  setup container        (runs once, verifies stack, exits)
        │
        │ healthy
        ▼
  agent container        (Python, LangGraph, Rocq/opam, PyGithub)
        │         │              │
        │         │              └──────────────────► GitHub
        │         │                                (PyGithub push)
        │         ▼
        │   SQLite container    (state, logs, metrics — persists)
        │
        ▼
  RL sandbox container   (generated RL code, yfinance, backtesting)
  └── isolated, no outbound network
  └── deployable anywhere as a standalone image
```

---

## Verification Architecture

This project uses **two independent verification layers**. They answer different questions and must not be confused.

### Layer 1 — Rocq (Formal Verification)
Answers: *does the code do what it claims?*

Rocq proves structural properties of the generated code, such as:
- Portfolio value never goes below zero
- Position sizes stay within defined bounds
- Reward function always returns a finite value
- State transitions are well-typed and deterministic

These are mathematical guarantees. If Rocq proves a property, it is unconditionally true.

### Layer 2 — Paper Trading Metrics (Empirical Verification)
Answers: *does the strategy actually perform well?*

Rocq cannot tell you if a strategy is profitable — that is statistical, not structural.
Performance is measured by running the agent against historical stock data and recording:

- **Total return** — overall profit/loss over the backtest period
- **Sharpe ratio** — return relative to risk (primary performance metric)
- **Max drawdown** — worst peak-to-trough loss
- **Win rate** — percentage of profitable trades
- **Cumulative reward** — RL training signal over episodes

Data source: `yfinance` (free, no account required)

---

## Technology Stack

### Host (native only)
| Component | Technology | Reason |
|---|---|---|
| LLM runtime | Ollama | Native host = direct CPU/GPU access, better performance than containerized |
| Model | `qwen2.5-coder:7b-instruct-q4_K_M` | Best coding performance within 8GB RAM |
| Quantization | Q4_K_M | ~4.5GB RAM footprint, good quality tradeoff |

> Ollama is the **only thing installed on the host**. Everything else runs in Docker.

### Docker Network
| Container | Technology | Responsibility |
|---|---|---|
| `setup` | Alpine + curl | Runs once — verifies Ollama reachable, SQLite writable, Rocq installed, GitHub token valid. Exits with pass/fail report. |
| `agent` | Python 3.11 + opam + Rocq | Runs LangGraph pipeline — researcher, coder, verifier agents |
| `sqlite` | SQLite via Python | Persists agent state, logs, and metrics across container rebuilds |
| `sandbox` | Python 3.11-slim | Runs agent-generated RL code in isolation — no outbound network, deployable standalone |

### Orchestration
| Component | Technology | Reason |
|---|---|---|
| Agent framework | LangGraph | Explicit graph control, supports conditional retry edges |
| Language | Python 3.11 | Best ecosystem for ML + agent tooling |
| Agent strategy | Single model, different system prompts per agent | Avoids multi-model RAM contention on 8GB RAM |

### Agent Tooling
| Component | Technology | Reason |
|---|---|---|
| Web search | Tavily API (free tier, 1000 req/month) | Structured results, easy Python integration |
| GitHub integration | PyGithub | Pure Python, no git binary required |
| Formal verification | Rocq via opam (inside agent container) | Industry standard proof assistant |

### Storage
| Component | Technology | Reason |
|---|---|---|
| State + logs + metrics | SQLite (dedicated container) | Zero config, persists across agent rebuilds |
| Generated code + proofs | Filesystem → GitHub | Plain `.py` and `.v` files, full audit trail |

### Paper Trading
| Component | Technology | Reason |
|---|---|---|
| Market data | `yfinance` | Free historical OHLCV, no account needed |
| RL framework | `gymnasium` + custom env | Standard RL interface, flexible reward shaping |
| Execution | Inside RL sandbox container | Isolated, portable, deployable to any machine |

---

## Container Communication

```
host:11434 (Ollama)
    ▲
    │ host.docker.internal:11434
    │
┌───┴──────────────────────────────────────┐
│  Docker network                           │
│                                           │
│  setup ──healthcheck──► agent             │
│                          │                │
│                          ├──► sqlite      │
│                          ├──► sandbox     │
│                          └──► GitHub      │
│                               (external)  │
└──────────────────────────────────────────┘
```

The `agent` container calls Ollama via `http://host.docker.internal:11434` — no LLM inference happens inside Docker. All other inter-service calls happen over the internal Docker network.

---

## Repository Structure

```
rl-trading-agent/
├── agents/
│   ├── researcher.py        # Tavily search, context gathering
│   ├── coder.py             # Code generation, handles retry context
│   └── verifier.py          # Rocq proof generation
├── tools/
│   ├── github_tool.py       # PyGithub commit/push helpers
│   ├── search_tool.py       # Tavily API wrapper
│   └── sandbox_tool.py      # Docker sandbox execution wrapper
├── env/
│   └── trading_env.py       # Gymnasium paper trading environment
├── metrics/
│   └── backtest.py          # Sharpe, drawdown, win rate calculations
├── proofs/                  # Generated Rocq .v files
├── output/                  # Generated Python code
├── docker/
│   ├── agent/
│   │   └── Dockerfile       # Python 3.11 + opam + Rocq + dependencies
│   ├── sandbox/
│   │   └── Dockerfile       # Python 3.11-slim, no outbound network
│   └── setup/
│       └── Dockerfile       # Alpine, healthcheck logic
├── db/
│   └── state.db             # SQLite — auto-created on first run
├── graph.py                 # LangGraph graph definition + edges
├── state.py                 # Shared state schema (TypedDict)
├── main.py                  # Entry point
├── docker-compose.yml       # Full stack definition
├── requirements.txt
└── .env                     # API keys — never committed
```

---

## Setup

### Prerequisites
Only two things needed on your host machine:

```bash
# 1. Install Docker
# https://docs.docker.com/engine/install/

# 2. Install Ollama and pull the model (~4.5GB download)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

### Environment Variables

```bash
# Copy the template
cp .env.example .env

# Fill in your values
TAVILY_API_KEY=your_key_here
GITHUB_TOKEN=your_personal_access_token
GITHUB_USERNAME=your_username
GITHUB_REPO=rl-trading-agent-output
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### Run

```bash
docker compose up
```

The setup container runs first and prints a health report before anything else starts:

```
[setup] checking Ollama at host.docker.internal:11434 ... PASS
[setup] checking SQLite volume writable             ... PASS
[setup] checking Rocq installation                  ... PASS
[setup] checking GitHub token                       ... PASS
[setup] all checks passed — starting agent
```

If any check fails the agent container does not start and the failure is printed clearly. Fix the issue and re-run `docker compose up`.

---

## Hardware Requirements

| Resource | Minimum | Notes |
|---|---|---|
| RAM | 8GB | Model ~4.5GB, agent stack ~1.5GB, ~2GB headroom |
| CPU | Intel i5 (any gen) | All LLM inference is CPU-bound via Ollama |
| GPU | Not required | Ollama uses CPU by default if VRAM is insufficient |
| Storage | 10GB free | Model ~4.5GB + Docker images ~2GB + generated code |
| OS | Any Docker-supported OS | Linux recommended; works on Windows/Mac via Docker Desktop |

---

## Key Design Decisions

**Ollama on host, everything else in Docker.** Native Ollama gives direct CPU/GPU access with no virtualization overhead — meaningfully faster inference on low-end hardware. All other components run in Docker so the host stays clean and the stack is fully portable with a single `docker compose up`.

**Single model for all agents.** Running multiple 7B models simultaneously would exceed 8GB RAM. Each agent gets a role-specific system prompt instead. The quality difference versus separate specialized models is negligible at PoC scale.

**Sequential agent execution.** Agents run one at a time, not in parallel. Keeps peak RAM usage predictable and makes the retry loop deterministic and easy to debug.

**Setup container with health report.** Rather than silently failing mid-run, a dedicated setup container verifies every dependency before the agent starts. Each check is reported individually so failures are immediately obvious.

**SQLite in a dedicated container.** Separating the database from the agent container means logs and metrics survive agent rebuilds and updates. The volume persists independently.

**RL sandbox is the deployable artifact.** The agent container is dev tooling — it stays on your machine. The sandbox container is what gets deployed. When a proof passes, the agent packages the verified RL code into the sandbox image and pushes it to GitHub. You can `docker run` it on any machine for paper trading.

**Rocq verifies structure, not profitability.** Formal verification is scoped to properties that can actually be proven — invariants, bounds, type safety. Trading performance is measured empirically via backtesting. These answer different questions and are complementary.

---

## Development Phases

### Phase 1 — Docker stack
- [ ] `docker-compose.yml` with all four containers defined
- [ ] Agent `Dockerfile` (Python 3.11 + opam + Rocq)
- [ ] Sandbox `Dockerfile` (Python 3.11-slim, isolated)
- [ ] Setup container healthcheck script
- [ ] SQLite schema defined
- [ ] `docker compose up` runs cleanly end to end

### Phase 2 — Core agent loop
- [ ] LangGraph state schema (`state.py`)
- [ ] Coder agent — generates RL trading code from prompt
- [ ] Verifier agent — generates Rocq proof for one property
- [ ] Retry edge — feeds Rocq failure back to coder
- [ ] Manual test of coder → verifier → retry loop

### Phase 3 — Full pipeline
- [ ] Researcher agent with Tavily search
- [ ] GitHub push on proof pass
- [ ] Paper trading environment (`trading_env.py`)
- [ ] Backtesting metrics logger writing to SQLite

### Phase 4 — Productionise
- [ ] Full end-to-end demo run
- [ ] SQLite logging for all agent decisions and retry counts
- [ ] RL sandbox image pushed to GitHub on verified pass
- [ ] README for the generated output repo

---

## Limitations (PoC Scope)

- Verifier agent generates Rocq proof stubs — complex proofs may require manual completion
- `yfinance` data is historical only, no real-time feed
- Single stock symbol per run to keep the RL environment simple
- No multi-GPU inference — all LLM calls are CPU-bound through Ollama

---

## License

MIT
