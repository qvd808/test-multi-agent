# MarginGuard — Verified RL Trading Sandbox

A Lean 4 powered reinforcement learning environment for paper trading. Every trade is mathematically verified by a formal specification before being executed in the Python sandbox.

---

## Architecture: The "Verified Shield"

1.  **Lean 4 (Referee)**: Defines the core invariants of trading (e.g., "Balance cannot go negative", "Cannot sell what you don't own").
2.  **Python (Sandbox)**: A Gymnasium-compatible environment that fetches live market data from `yfinance`.
3.  **The Bridge**: Every `step()` in the RL environment calls the compiled Lean binary to validate the action. If Lean "vetoes" the trade, the state is preserved and the agent receives a massive penalty.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install gymnasium yfinance numpy
```

### 2. Build the Verified Core
```bash
cd proofs
lake build
cd ..
```

### 3. Run the Sandbox Test
Verify that the Lean shield is active and protecting the portfolio:
```bash
python3 env/paper_env.py
```

---

## Project Structure

- `proofs/`: Lean 4 formal specification (The "Source of Truth").
- `env/`: Python Gymnasium wrapper with Lean FFI/Pipe integration.
- `docker/`: Containerized verification environments.

---

## Invariants Protected by Lean
- [x] **No Naked Shorting**: Quantity to sell must be <= current position.
- [x] **Solvency**: Purchase cost must be <= current balance.
- [x] **State Preservation**: Illegal trades result in zero state change (Veto).
- [x] **Verification Loop**: Agent is penalized for hitting the formal shield.
