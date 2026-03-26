import gymnasium as gym
from stable_baselines3 import PPO
from env.paper_env import MarginGuardEnv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def run_visualization(model_path="margin_guard_pro_v3_ppo.zip", steps=200):
    """
    Diagnostic visualization of a trained agent's performance.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the agent first.")
        return

    # 1. Setup Env and Model
    env   = MarginGuardEnv(ticker="ETH-USD", history_length=5, use_cache=True)
    model = PPO.load(model_path, env=env)
    
    obs, _ = env.reset()
    
    history = {
        "price":     [],
        "balance":   [],
        "position":  [],
        "portfolio": [],
        "reward":    [],
        "action":    [],
        "vetoed":    [],
    }

    # 2. Sequential simulation
    print(f"[vis] Running simulation for {steps} steps...")
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        history["price"].append(info["price"])
        history["balance"].append(info["balance"])
        history["position"].append(info["position"])
        history["portfolio"].append(info["balance"] + info["position"] * info["price"])
        history["reward"].append(reward)
        history["action"].append(info["action"])
        history["vetoed"].append(info["vetoed"])
        
        if done: break

    # 3. Create 6-panel Dashboard
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
    fig.suptitle(f"MarginGuard v3 — Diagnostic Dashboard (ETH-USD)", fontsize=16, fontweight='bold')

    # Panel 1: Price and Trades
    axes[0, 0].plot(df["price"], color="gray", alpha=0.5, label="Price")
    # Mark buys/sells
    buys = df[df["action"] > 0]
    sells = df[df["action"] < 0]
    axes[0, 0].scatter(buys.index, buys["price"], color="green", marker="^", label="Buy", s=50)
    axes[0, 0].scatter(sells.index, sells["price"], color="red", marker="v", label="Sell", s=50)
    axes[0, 0].set_title("1. Market Price & Execution")
    axes[0, 0].legend()

    # Panel 2: Portfolio Value & Drawdown
    axes[0, 1].plot(df["portfolio"], color="blue", label="Portfolio Value")
    # Drawdown calculation
    rolling_max = df["portfolio"].cummax()
    drawdown = (df["portfolio"] - rolling_max) / rolling_max
    ax2 = axes[0, 1].twinx()
    ax2.fill_between(df.index, drawdown, color="red", alpha=0.1, label="Drawdown")
    axes[0, 1].set_title("2. Portfolio Value & Drawdown (%)")
    axes[0, 1].set_ylabel("Value ($)")
    ax2.set_ylabel("Drawdown (%)")

    # Panel 3: Asset Position
    axes[1, 0].step(df.index, df["position"], color="purple", where="post")
    axes[1, 0].axhline(0, color="black", linestyle="--", alpha=0.3)
    axes[1, 0].set_title("3. Asset Position Size")
    axes[1, 0].set_ylabel("Quantity (Lots)")

    # Panel 4: Step Reward (Verified)
    axes[1, 1].bar(df.index, df["reward"], color="teal", alpha=0.6)
    axes[1, 1].set_title("4. Verified Step Reward (Lean Core)")
    axes[1, 1].set_ylabel("Reward Units")

    # Panel 5: Action Distribution (Histogram)
    axes[2, 0].hist(df["action"], bins=21, color="orange", alpha=0.7, edgecolor="black")
    axes[2, 0].set_title("5. Action Distribution (Lots)")
    axes[2, 0].set_xlabel("Action Qty")

    # Panel 6: Reward Convergence (Cumulative)
    axes[2, 1].plot(df["reward"].cumsum(), color="gold", linewidth=2)
    axes[2, 1].set_title("6. Cumulative Verified Reward")
    axes[2, 1].set_ylabel("Total Reward")

    # Save to artifacts directory
    output_path = "/root/.gemini/antigravity/brain/595ad11d-e7ec-432f-90d6-28180382fb88/performance_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[vis] Dashboard saved to {output_path}")

if __name__ == "__main__":
    run_visualization()
