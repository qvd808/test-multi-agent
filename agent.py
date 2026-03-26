import os
import sys
import gymnasium as gym
import numpy as np

# Ensure the local environment can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

try:
    from stable_baselines3 import PPO
    from env.paper_env import MarginGuardEnv
except ImportError:
    print("Dependencies missing. Run: pip install gymnasium yfinance numpy stable-baselines3[extra] shimmy>=2.0")
    sys.exit(1)

def train_agent():
    """
    Trains a PPO agent in the Pro-Enhanced MarginGuard sandbox.
    """
    print("--- MarginGuard 'Pro' Training Start ---")
    
    # 1. Instantiate the Pro Environment
    # history_length=5, use_cache=True (Yahoo Finance Protection)
    try:
        env = MarginGuardEnv(ticker="NVDA", initial_balance=50000, history_length=5, use_cache=True)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # 2. Initialize the PPO Model
    # We now have 7-dimensional normalized observations
    print(f"[agent] Observation Space: {env.observation_space}")
    print("[agent] Building PPO model...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=1e-4, # Lower LR for stability with normalized inputs
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log="./margin_guard_logs/"
    )

    # 3. Train for 100,000 timesteps
    # Watch ep_rew_mean climb as it learns the Lean Shield rules
    print("[agent] Starting training... (Cached data helps avoid rate limits)")
    model.learn(total_timesteps=100000, progress_bar=True)

    # 4. Save the Model
    model.save("margin_guard_pro_ppo")
    print("[agent] Pro Model saved to margin_guard_pro_ppo.zip")

def evaluate_agent():
    print("\n--- Starting Pro Evaluation Loop (10 Steps) ---")
    env = MarginGuardEnv(ticker="NVDA", initial_balance=50000, history_length=5, use_cache=True)
    model = PPO.load("margin_guard_pro_ppo")

    obs, _ = env.reset()
    for i in range(1, 11):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Un-normalize for display
        balance = obs[0] * 50000
        position = obs[1] * 100
        
        print(f"Step {i}:")
        print(f"  Action (Qty): {action}")
        print(f"  Obs -> Bal: ${balance:.2f}, Pos: {position:.2f}")
        print(f"  Step Reward: {reward:.2f}")
        
        if terminated or truncated:
            break

if __name__ == "__main__":
    train_agent()
    evaluate_agent()
