import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yfinance as yf
import subprocess
import os

class MarginGuardEnv(gym.Env):
    """
    A live paper-trading RL environment where state transitions 
    are mathematically verified by a Lean 4 compiled binary via Pipe interface.
    """
    def __init__(self, ticker="AAPL", initial_balance=100000, binary_path="./proofs/.lake/build/bin/margin_proofs"):
        super().__init__()
        
        # Real-time data source
        self.ticker_name = ticker
        self.ticker = yf.Ticker(ticker)
        self.initial_balance = initial_balance
        self.binary_path = binary_path
        
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"Missing {self.binary_path}. Run 'lake build' in the proofs directory.")
            
        # 1. Action Space: Sell up to 10 shares (-10), Buy up to 10 shares (+10)
        self.action_space = spaces.Discrete(21, start=-10)
        
        # 2. Observation Space: [Balance, Position, Current Price]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.int32)

    def _consult_lean_shield(self, b, p, price, q):
        """Ask the Lean Referee to evaluate the trade."""
        input_str = f"{int(b)} {int(p)} {int(price)} {int(q)}\n"
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
        return int(parts[0]), int(parts[1]), int(parts[2])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0
        
        # Fetch initial market price
        fast_info = self.ticker.fast_info
        current_price = int(fast_info.get('lastPrice', 150))
        
        # Track portfolio value across time
        self.prev_portfolio_value = self.balance + (self.position * current_price)
        
        return np.array([self.balance, self.position, current_price], dtype=np.int32), {}

    def step(self, action_qty):
        # 1. Get Live Market Data
        try:
            current_price = int(self.ticker.fast_info['lastPrice'])
        except Exception:
            current_price = 150 # Fallback for demo
        
        # 2. Lean Referee Evaluates the Trade via Pipe
        new_balance, new_position, lean_reward = self._consult_lean_shield(
            self.balance, self.position, current_price, action_qty
        )

        # 3. Calculate Final RL Reward
        if lean_reward == -1000:
            final_reward = lean_reward 
        else:
            # Calculate the REAL portfolio value at the current market price
            current_portfolio_value = new_balance + (new_position * current_price)
            
            # The reward is how much money we made/lost since the LAST step
            final_reward = current_portfolio_value - self.prev_portfolio_value
            
            # Update our tracker for the next step
            self.prev_portfolio_value = current_portfolio_value

        # 4. Commit the verified state
        self.balance = new_balance
        self.position = new_position
        
        obs = np.array([self.balance, self.position, current_price], dtype=np.int32)
        done = self.balance <= 0 # End episode if bankrupt
        
        return obs, final_reward, done, False, {}

# Quick Test Execution
if __name__ == "__main__":
    env = MarginGuardEnv(ticker="NVDA", initial_balance=50000)
    obs, _ = env.reset()
    print(f"Initial State -> Balance: ${env.balance}, Position: {env.position}, NVDA Price: ${obs[2]}")
    
    # Simulate an agent trying to buy 5 shares
    print("\n--- Agent attempts to BUY 5 shares ---")
    next_obs, reward, done, _, _ = env.step(5)
    print(f"New State -> Balance: ${next_obs[0]}, Position: {next_obs[1]}")
    print(f"Reward Issued: {reward}")

    # Simulate an agent hallucinating and trying to short 100 shares it doesn't own
    print("\n--- Agent attempts to SELL 100 shares (Illegal) ---")
    next_obs, reward, done, _, _ = env.step(-100)
    print(f"New State -> Balance: ${next_obs[0]}, Position: {next_obs[1]} (Notice it didn't change!)")
    print(f"Reward Issued: {reward} (Lean Shield Penalty applied)")
