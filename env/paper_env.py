import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yfinance as yf
import subprocess
import os
from collections import deque
from datetime import datetime

class MarginGuardEnv(gym.Env):
    """
    MarginGuard — Lean Interface Layer

    Design principle: This file is a TRANSPORT LAYER, not a reward designer.
    All reward logic lives in the Lean binary. Python's only jobs are:
      1. Fetch price data
      2. Translate action index → lot qty
      3. Serialize state → Lean binary (stdin)
      4. Deserialize Lean output (stdout) → gym step return
      5. Build the observation vector

    Lean binary I/O contract (all values in integer cents):
      stdin : "<balance_cents> <position> <price_cents> <qty> <prev_price_cents>"
      stdout: "<new_balance_cents> <new_position> <reward_cents>"

    The extra `prev_price_cents` field lets Lean compute momentum and holding
    penalties itself, keeping all reward math formally verified.
    """

    MAX_STEPS = 500  # Episode cap

    def __init__(
        self,
        ticker="ETH-USD",
        initial_balance=50_000,
        history_length=5,
        use_cache=True,
        binary_path="./proofs/.lake/build/bin/margin_proofs",
    ):
        super().__init__()

        self.ticker_name     = ticker
        self.initial_balance = initial_balance
        self.history_length  = history_length
        self.binary_path      = binary_path

        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(
                f"Lean binary not found: {self.binary_path}\n"
                f"Run 'lake build' inside the proofs/ directory."
            )

        # ── Data cache ────────────────────────────────────────────────────────
        self.use_cache   = use_cache
        self.data_buffer = []
        self.current_idx = 0

        if self.use_cache:
            print(f"[env] Fetching {ticker} historical data...")
            df = yf.download(ticker, period="1y", interval="1h", progress=False)
            self.data_buffer = df["Close"].values.flatten().tolist()
            print(f"[env] Cached {len(self.data_buffer)} hourly bars.")

        # ── Action space ──────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(21)
        self.ACTION_MAP   = list(range(-10, 11))

        # ── Observation space ─────────────────────────────────────────────────
        obs_size = 2 + self.history_length
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.price_history = deque(maxlen=self.history_length + 1)
        self._steps = 0

    # ── Lean IPC ──────────────────────────────────────────────────────────────

    def _consult_lean(self, balance, position, price, qty, prev_price):
        """
        Single call to the Lean binary.
        """
        payload = (
            f"{int(balance * 100)} "
            f"{int(position)} "
            f"{int(price * 100)} "
            f"{int(qty)} "
            f"{int(prev_price * 100)}\n"
        )
        proc = subprocess.Popen(
            [self.binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(input=payload)
        if proc.returncode != 0:
            raise RuntimeError(f"Lean shield error:\n{stderr.strip()}")

        parts = stdout.strip().split()
        new_balance  = int(parts[0]) / 100   # cents → dollars
        new_position = int(parts[1])
        reward       = int(parts[2]) / 100   # cents → reward units
        return new_balance, new_position, reward

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _get_price(self):
        if self.use_cache and self.current_idx < len(self.data_buffer):
            p = float(self.data_buffer[self.current_idx])
            self.current_idx += 1
            return p
        if self.use_cache:
            self.current_idx = 0
        return float(yf.Ticker(self.ticker_name).fast_info.get("lastPrice", 2000))

    def _build_obs(self):
        norm_bal = self.balance / self.initial_balance
        norm_pos = np.clip(self.position / 10.0, -1.0, 1.0)

        prices = list(self.price_history)
        ratios = [
            prices[i] / prices[i - 1] if prices[i - 1] != 0 else 1.0
            for i in range(1, len(prices))
        ]
        while len(ratios) < self.history_length:
            ratios.insert(0, 1.0)

        return np.array([norm_bal, norm_pos] + ratios, dtype=np.float32)

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance  = self.initial_balance
        self.position = 0
        self._steps   = 0

        if self.use_cache:
            self.current_idx = np.random.randint(
                0, max(1, len(self.data_buffer) - self.MAX_STEPS - 10)
            )

        price = self._get_price()
        self.price_history.clear()
        self.price_history.append(price)

        return self._build_obs(), {}

    def step(self, action_idx):
        self._steps += 1
        action_qty = self.ACTION_MAP[int(action_idx)]

        prev_price    = float(self.price_history[-1])
        current_price = self._get_price()
        self.price_history.append(current_price)

        # ── Entire reward computation happens inside Lean ─────────────────────
        new_balance, new_position, reward = self._consult_lean(
            self.balance, self.position,
            current_price, action_qty,
            prev_price,
        )

        # ── Log trades ────────────────────────────────────────────────────────
        if action_qty != 0:
            side   = "BUY " if action_qty > 0 else "SELL"
            vetoed = (new_balance == self.balance and new_position == self.position)
            status = "VETOED " if vetoed else "SUCCESS"
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] {side} {abs(action_qty):2d}"
                f" @ ${current_price:,.2f} | {status} | reward={reward:+.4f}"
            )

        self.balance  = new_balance
        self.position = new_position

        obs  = self._build_obs()
        done = (
            self.balance <= 0
            or (self.use_cache and self.current_idx >= len(self.data_buffer))
            or self._steps >= self.MAX_STEPS
        )

        info = {
            "price":    current_price,
            "balance":  new_balance,
            "position": new_position,
            "action":   action_qty,
            "reward":   reward,
            "vetoed":   (new_balance == self.balance and new_position == self.position and action_qty != 0)
        }

        return obs, float(reward), done, False, info

if __name__ == "__main__":
    env = MarginGuardEnv(ticker="ETH-USD", history_length=5)
    obs, _ = env.reset()
    print(f"Initial observation (size {len(obs)}): {obs}\n")

    print("--- Simulating 5 steps ---")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i+1} | action={env.ACTION_MAP[action]:+3d} | reward={reward:+.2f}")
        if done: break