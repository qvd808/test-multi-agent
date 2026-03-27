# env/paper_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yfinance as yf
import os
from collections import deque
from datetime import datetime

# Import FFI (auto-builds if needed)
try:
    from leanffi import MarginGuardCore
    USE_FFI = True
    print("[env] Using FFI for Lean calls")
except Exception as e:
    print(f"[env] FFI import failed: {e}")
    USE_FFI = False
    raise  # Don't fall back - fix the build instead

class MarginGuardEnv(gym.Env):
    MAX_STEPS = 500

    def __init__(
        self,
        ticker="ETH-USD",
        initial_balance=50_000,
        history_length=5,
        use_cache=True,
        binary_path=None,  # Not used with FFI
    ):
        super().__init__()

        self.ticker_name = ticker
        self.initial_balance = initial_balance
        self.history_length = history_length

        # Initialize FFI core (auto-builds if needed)
        self.core = MarginGuardCore()
        print(f"[env] Lean verified core initialized")

        # ── Data cache ────────────────────────────────────────────────────────
        self.use_cache = use_cache
        self.data_buffer = []
        self.current_idx = 0

        if self.use_cache:
            print(f"[env] Fetching {ticker} historical data...")
            df = yf.download(ticker, period="1y", interval="1h", progress=False)
            self.data_buffer = df["Close"].values.flatten().tolist()
            print(f"[env] Cached {len(self.data_buffer)} hourly bars.")

        # ── Action/Observation spaces ─────────────────────────────────────────
        self.action_space = spaces.Discrete(21)
        self.ACTION_MAP = list(range(-10, 11))

        self.price_history = deque(maxlen=self.history_length + 1)
        self.avg_entry_price = 0.0
        self._steps = 0

        # ── Expanded Observation Space ────────────────────────────────────────
        # [Bal, Pos, Week Signal, Entry Ratio, Ratio1, Ratio2, Ratio3, Ratio4, Ratio5]
        obs_size = 4 + self.history_length
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _consult_lean(self, balance, position, price, qty, prev_price, avg_entry, sma_week):
        """Fast FFI call to verified Lean functions"""
        return self.core.trade(balance, position, price, qty, prev_price, avg_entry, sma_week)

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

        # 1-Week Signal (168 hours)
        current_price = self.price_history[-1]
        if self.use_cache and self.current_idx >= 168:
            week_data = self.data_buffer[self.current_idx-168:self.current_idx]
            week_sma = sum(week_data) / len(week_data)
        else:
            week_sma = current_price
            
        week_signal = current_price / week_sma if week_sma > 0 else 1.0
        entry_ratio = self.avg_entry_price / current_price if self.avg_entry_price > 0 else 0.0

        prices = list(self.price_history)
        ratios = [
            prices[i] / prices[i - 1] if prices[i - 1] != 0 else 1.0
            for i in range(1, len(prices))
        ]
        while len(ratios) < self.history_length:
            ratios.insert(0, 1.0)

        return np.array([norm_bal, norm_pos, week_signal, entry_ratio] + ratios, dtype=np.float32), week_sma

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0
        self.avg_entry_price = 0.0
        self._steps = 0

        if self.use_cache:
            self.current_idx = np.random.randint(
                0, max(1, len(self.data_buffer) - self.MAX_STEPS - 10)
            )

        price = self._get_price()
        self.price_history.clear()
        self.price_history.append(price)

        obs, _ = self._build_obs()
        return obs, {}

    def step(self, action_idx):
        self._steps += 1
        action_qty = self.ACTION_MAP[int(action_idx)]

        prev_price = float(self.price_history[-1])
        current_price = self._get_price()
        self.price_history.append(current_price)

        obs, sma_week = self._build_obs()

        # Fast FFI call with strategy markers
        new_balance, new_position, reward = self._consult_lean(
            self.balance, self.position,
            current_price, action_qty,
            prev_price, self.avg_entry_price, sma_week
        )

        # Update Cost Basis
        if not (abs(new_balance - self.balance) < 0.01 and abs(new_position - self.position) < 0.01):
            if action_qty > 0: # Buy
                total_cost = (self.position * self.avg_entry_price) + (action_qty * current_price)
                self.avg_entry_price = total_cost / (self.position + action_qty)
            
        if new_position == 0:
            self.avg_entry_price = 0.0

        if action_qty != 0:
            side = "BUY " if action_qty > 0 else "SELL"
            vetoed = abs(new_balance - self.balance) < 0.01 and abs(new_position - self.position) < 0.01
            status = "VETOED " if vetoed else "SUCCESS"
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] {side} {abs(action_qty):2d}"
                f" @ ${current_price:,.2f} | {status} | reward={reward:+.4f} | Entry=${self.avg_entry_price:,.2f}"
            )

        self.balance = new_balance
        self.position = new_position

        done = (
            self.balance <= 0
            or (self.use_cache and self.current_idx >= len(self.data_buffer))
            or self._steps >= self.MAX_STEPS
        )

        info = {
            "price": current_price,
            "balance": new_balance,
            "position": new_position,
            "action": action_qty,
            "reward": reward,
            "vetoed": (abs(new_balance - self.balance) < 0.01 and abs(new_position - self.position) < 0.01 and action_qty != 0)
        }

        return obs, float(reward), done, False, info

if __name__ == "__main__":
    env = MarginGuardEnv(ticker="ETH-USD", history_length=5)
    obs, _ = env.reset()
    print(f"Initial observation (size {len(obs)}): {obs}\n")

    print("--- Testing 5 steps ---")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i+1} | action={env.ACTION_MAP[action]:+3d} | reward={reward:+.2f} | balance={info['balance']:.2f}")
        if done:
            break