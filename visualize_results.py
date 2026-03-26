import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.paper_env import MarginGuardEnv

def run_visualization(steps=200):
    print(f"--- Visualizing MarginGuard PPO Behavior ({steps} steps) ---")
    
    # 1. Initialize Env and Load Model
    env = MarginGuardEnv(ticker="ETH-USD", initial_balance=50000, history_length=5, use_cache=True)
    model = PPO.load("margin_guard_pro_ppo")
    
    # Data containers
    prices = []
    portfolio_values = []
    actions = []
    balances = []
    positions = []
    timestamps = range(steps)
    
    # 2. Run Simulation
    obs, _ = env.reset()
    for i in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        
        # Capture raw price before step
        # Note: obs structure is [norm_bal, norm_pos, ratio1, ...]
        # We'll get the real price from the env history
        current_price = env.price_history[-1]
        portfolio_val = env.balance + (env.position * current_price)
        
        prices.append(current_price)
        portfolio_values.append(portfolio_val)
        actions.append(action)
        balances.append(env.balance)
        positions.append(env.position)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            print(f"Simulation ended early at step {i}")
            timestamps = range(len(prices))
            break

    # 3. Create Multi-Panel Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # --- Panel 1: Price and Trade Markers ---
    ax1.plot(timestamps, prices, label='ETH-USD Price', color='blue', alpha=0.6)
    
    # Plot Buys (Action > 0)
    buy_indices = [i for i, a in enumerate(actions) if a > 0]
    ax1.scatter(buy_indices, [prices[i] for i in buy_indices], marker='^', color='green', label='BUY', s=50, zorder=5)
    
    # Plot Sells (Action < 0)
    sell_indices = [i for i, a in enumerate(actions) if a < 0]
    ax1.scatter(sell_indices, [prices[i] for i in sell_indices], marker='v', color='red', label='SELL', s=50, zorder=5)
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title('MarginGuard Trade Decisions vs Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: Portfolio Value ---
    ax2.plot(timestamps, portfolio_values, label='Portfolio Value (Total Money)', color='purple', linewidth=2)
    ax2.axhline(y=50000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax2.set_ylabel('Total Value ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Panel 3: Position Size ---
    ax3.bar(timestamps, positions, color='orange', alpha=0.7, label='Current Position (Shares)')
    ax3.set_ylabel('Shares')
    ax3.set_xlabel('Time Step (Hourly)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the output
    save_path = "performance_plot.png"
    plt.savefig(save_path)
    print(f"Plot saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    run_visualization()
