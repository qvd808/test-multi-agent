import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yfinance as yf
import subprocess
import os
from collections import deque

class MarginGuardEnv(gym.Env):
    """
    A pro-grade paper-trading RL environment with:
    1. Lean 4 Formal Verification (The Shield)
    2. Observation History (Trailing Prices)
    3. Input Normalization (For Deep Learning stability)
    4. Historical Data Caching (To avoid yfinance throttling)
    """
    def __init__(self, ticker="AAPL", initial_balance=100000, 
                 history_length=5, use_cache=True,
                 binary_path="./proofs/.lake/build/bin/margin_proofs"):
        super().__init__()
        
        self.ticker_name = ticker
        self.initial_balance = initial_balance
        self.history_length = history_length
        self.binary_path = binary_path
        
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"Missing {self.binary_path}. Run 'lake build' in proofs/.")
            
        # 1. Data Caching (yfinance protection)
        self.use_cache = use_cache
        self.data_buffer = []
        self.current_idx = 0
        if self.use_cache:
            print(f"[env] Fetching historical data for {ticker}...")
            df = yf.download(ticker, period="1y", interval="1h", progress=False)
            self.data_buffer = df['Close'].values.flatten().tolist()
            print(f"[env] Cached {len(self.data_buffer)} data points.")

        # 2. Action Space: Discrete(21) -> [-10, +10]
        self.action_space = spaces.Discrete(21, start=-10)
        
        # 3. Observation Space: [norm_bal, norm_pos, price_ratio_1, ..., price_ratio_N]
        # Size = 2 + history_length
        obs_size = 2 + self.history_length
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        self.price_history = deque(maxlen=self.history_length + 1)

    def _consult_lean_shield(self, b, p, price, q):
        # Lean expects monetary values in cents, so balance and price are multiplied by 100
        input_str = f"{int(b * 100)} {int(p)} {int(price)} {int(q)}\n"
        process = subprocess.Popen(
            [self.binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_str)
        if process.returncode != 0:
            raise Exception(f"Lean Shield Error: {stderr}")
        parts = stdout.strip().split(" ")
        # Lean returns new_balance in cents, so divide by 100 to convert back to dollars
        return int(parts[0]) / 100, int(parts[1]), int(parts[2])

    def _get_price(self):
        if self.use_cache and self.current_idx < len(self.data_buffer):
            price = self.data_buffer[self.current_idx]
            self.current_idx += 1
            return price
        else:
            # Fallback to live data or restart cache
            if self.use_cache: self.current_idx = 0
            ticker = yf.Ticker(self.ticker_name)
            return ticker.fast_info.get('lastPrice', 150)

    def _get_obs(self):
        # Normalization:
        # 1. Balance relative to initial
        norm_bal = self.balance / self.initial_balance
        # 2. Position relative to a safe 'unit' (e.g. 100 shares)
        norm_pos = self.position / 100.0
        
        # 3. Price History Ratios (p[t] / p[t-1])
        # This gives the agent relative movement which is stationary
        price_ratios = []
        prices = list(self.price_history)
        for i in range(1, len(prices)):
            ratio = prices[i] / prices[i-1] if prices[i-1] != 0 else 1.0
            price_ratios.append(ratio)
            
        # If history isn't full yet, pad with 1.0
        while len(price_ratios) < self.history_length:
            price_ratios.insert(0, 1.0)
            
        return np.array([norm_bal, norm_pos] + price_ratios, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0
        if self.use_cache: self.current_idx = np.random.randint(0, max(1, len(self.data_buffer) - 100))
        
        current_price = self._get_price()
        self.price_history.clear()
        self.price_history.append(current_price)
        
        # Portfolio value is balance (dollars) + position * price (dollars)
        self.prev_portfolio_value = self.balance + (self.position * current_price)
        return self._get_obs(), {}

    def step(self, action_qty):
        current_price = self._get_price()
        self.price_history.append(current_price)
        
        # Lean Referee evaluates the trade
        # current_price is multiplied by 100 to convert to cents for Lean
        new_balance, new_position, lean_reward = self._consult_lean_shield(
            self.balance, self.position, int(current_price * 100), action_qty
        )

        if lean_reward == -1000:
            final_reward = lean_reward 
        else:
            # current_portfolio_value is balance (dollars) + position * price (dollars)
            current_portfolio_value = new_balance + (new_position * current_price)
            final_reward = current_portfolio_value - self.prev_portfolio_value
            self.prev_portfolio_value = current_portfolio_value

        self.balance = new_balance
        self.position = new_position
        
        obs = self._get_obs()
        done = self.balance <= 0 or (self.use_cache and self.current_idx >= len(self.data_buffer))
        
        return obs, float(final_reward), done, False, {}

if __name__ == "__main__":
    env = MarginGuardEnv(ticker="NVDA", history_length=5)
    obs, _ = env.reset()
    print(f"Initial Pro-Observation (Size {len(obs)}): {obs}")
    
    print("\n--- Simulating 5 steps ---")
    for i in range(5):
        obs, reward, done, _, _ = env.step(1)
        print(f"Step {i+1} Reward: {reward:.2f} | Obs[0:2]: {obs[0:2]}")
