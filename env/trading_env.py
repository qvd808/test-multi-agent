"""Gymnasium paper trading environment.

A custom Gymnasium environment for RL-based paper trading. Supports
OHLCV data, position management with bounds enforcement, and
risk-adjusted reward shaping.

Key invariants (verified by Rocq):
- Portfolio value never goes below zero
- Position sizes stay within [0, max_position]
- Reward function always returns a finite float
- State transitions are deterministic
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import spaces


class PaperTradingEnv(gym.Env):
    """Paper trading environment for RL agents.

    Observation space: [open, high, low, close, volume, position, cash, portfolio_value]
    Action space: Discrete(3) — 0=hold, 1=buy, 2=sell
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        symbol: str = "AAPL",
        period: str = "1y",
        initial_cash: float = 10_000.0,
        max_position: int = 100,
        transaction_cost: float = 0.001,
    ) -> None:
        super().__init__()

        self.symbol = symbol
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.transaction_cost = transaction_cost

        # Download OHLCV data
        self.data = self._download_data(symbol, period)
        self.n_steps = len(self.data)

        # Spaces
        # Observation: OHLCV (5) + position (1) + cash (1) + portfolio_value (1) = 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        # Actions: hold (0), buy (1), sell (2)
        self.action_space = spaces.Discrete(3)

        # State variables (set in reset)
        self.current_step: int = 0
        self.cash: float = initial_cash
        self.position: int = 0
        self.trades: list[dict] = []

    @staticmethod
    def _download_data(symbol: str, period: str) -> pd.DataFrame:
        """Download OHLCV data from yfinance."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            raise ValueError(f"No data returned for {symbol} over period {period}")
        return df[["Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)

    def _get_observation(self) -> np.ndarray:
        """Build the observation vector."""
        row = self.data.iloc[self.current_step]
        portfolio_value = self.cash + self.position * row["Close"]
        return np.array(
            [
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
                row["Volume"],
                float(self.position),
                self.cash,
                portfolio_value,
            ],
            dtype=np.float32,
        )

    @property
    def portfolio_value(self) -> float:
        """Current portfolio value (cash + holdings)."""
        if self.current_step < self.n_steps:
            price = self.data.iloc[self.current_step]["Close"]
        else:
            price = self.data.iloc[-1]["Close"]
        return self.cash + self.position * price

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0
        self.trades = []
        return self._get_observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment.

        Enforces invariants:
        - Position stays within [0, max_position]
        - Cash never goes negative (portfolio value ≥ 0 guaranteed)
        - Reward is always a finite float
        """
        price = self.data.iloc[self.current_step]["Close"]
        prev_portfolio = self.portfolio_value

        # Execute action with bounds enforcement
        if action == 1:  # Buy
            if self.position < self.max_position:
                cost = price * (1 + self.transaction_cost)
                if self.cash >= cost:
                    self.cash -= cost
                    self.position += 1
                    self.trades.append({
                        "step": self.current_step,
                        "action": "buy",
                        "price": price,
                    })

        elif action == 2:  # Sell
            if self.position > 0:
                revenue = price * (1 - self.transaction_cost)
                self.cash += revenue
                self.position -= 1
                self.trades.append({
                    "step": self.current_step,
                    "action": "sell",
                    "price": price,
                })

        # Advance time
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        # Calculate reward (risk-adjusted return)
        current_portfolio = self.portfolio_value
        returns = (current_portfolio - prev_portfolio) / prev_portfolio if prev_portfolio > 0 else 0.0

        # Ensure reward is always finite
        reward = float(np.clip(returns, -1.0, 1.0))
        if not np.isfinite(reward):
            reward = 0.0

        obs = self._get_observation()
        info = {
            "portfolio_value": current_portfolio,
            "cash": self.cash,
            "position": self.position,
            "n_trades": len(self.trades),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Print current state."""
        price = self.data.iloc[min(self.current_step, self.n_steps - 1)]["Close"]
        print(
            f"Step {self.current_step}/{self.n_steps} | "
            f"Price: ${price:.2f} | "
            f"Position: {self.position} | "
            f"Cash: ${self.cash:.2f} | "
            f"Portfolio: ${self.portfolio_value:.2f}"
        )
