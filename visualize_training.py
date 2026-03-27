import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def visualize_training(csv_path="training_history_high_res.csv", save_path="training_performance.png"):
    if not os.path.exists(csv_path):
        fallback = "training_history_audit.csv"
        if os.path.exists(fallback):
            print(f"Warning: {csv_path} not found. Using fallback: {fallback}")
            csv_path = fallback
        else:
            print(f"Error: No training data found at {csv_path} or {fallback}.")
            return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: CSV is empty.")
        return

    # Check for columns and normalize naming
    # New format: step, price, action, vetoed
    # Old format: env/reward_step, env/price, env/action, env/is_vetoed
    if "env/price" in df.columns:
        df = df.rename(columns={
            "env/reward_step": "step",
            "env/price": "price",
            "env/action": "action",
            "env/is_vetoed": "vetoed",
            "env/balance": "balance",
            "env/reward": "reward"
        })

    # Ensure necessary columns exist
    required = ["step", "price", "action", "vetoed", "balance", "reward"]
    for col in required:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}' in data.")
            return

    # ── Downsampling for Performance ──────────────────────────────────────────
    # If the file is huge (e.g. > 100k steps), downsample for plotting speed
    MAX_POINTS = 10000
    if len(df) > MAX_POINTS:
        factor = len(df) // MAX_POINTS
        print(f"Downsampling data by 1/{factor} for plotting performance.")
        # But we want to keep all BUY/SELL/VETO events for accuracy
        event_mask = (df["action"] != 0) | (df["vetoed"] == True)
        df_downsampled = df.iloc[::factor, :]
        df_events = df[event_mask]
        # Combine and sort to ensure we don't miss trades
        df_plot = pd.concat([df_downsampled, df_events]).drop_duplicates().sort_values("step")
    else:
        df_plot = df

    fig, axes = plt.subplots(5, 1, figsize=(14, 25), sharex=True)
    steps = df_plot["step"]

    # 1. Market Price
    axes[0].plot(steps, df_plot["price"], color="gray", alpha=0.3, label="Price")
    axes[0].set_title("1. Market Price (Detailed History)", fontsize=14)
    axes[0].set_ylabel("Price (USD)")

    # 2. BUY Markers
    axes[1].plot(steps, df_plot["price"], color="gray", alpha=0.2) # Background Price
    buy_mask = (df_plot["action"] > 0) & (df_plot["vetoed"] == False)
    if buy_mask.any():
        axes[1].scatter(steps[buy_mask], df_plot["price"][buy_mask], marker="^", color="green", s=60, label="BUY")
    axes[1].set_title("2. Executed BUY Events (w/ Market Context)", fontsize=14)
    axes[1].set_ylabel("Price")

    # 3. SELL Markers
    axes[2].plot(steps, df_plot["price"], color="gray", alpha=0.2) # Background Price
    sell_mask = (df_plot["action"] < 0) & (df_plot["vetoed"] == False)
    if sell_mask.any():
        axes[2].scatter(steps[sell_mask], df_plot["price"][sell_mask], marker="v", color="red", s=60, label="SELL")
    axes[2].set_title("3. Executed SELL Events (w/ Market Context)", fontsize=14)
    axes[2].set_ylabel("Price")

    # 4. VETO Markers
    axes[3].plot(steps, df_plot["price"], color="gray", alpha=0.2) # Background Price
    # Note: 'vetoed' might be boolean or int (0/1)
    veto_mask = (df_plot["vetoed"] == True) | (df_plot["vetoed"] == 1)
    if veto_mask.any():
        axes[3].scatter(steps[veto_mask], df_plot["price"][veto_mask], marker="x", color="black", s=50, label="VETO")
    axes[3].set_title("4. VETO Events (Lean Guaranteed)", fontsize=14)
    axes[3].set_ylabel("Price")

    # 5. Performance
    axes[4].plot(steps, df_plot["balance"], color="blue", label="Balance")
    ax5_reward = axes[4].twinx()
    ax5_reward.plot(steps, df_plot["reward"].cumsum(), color="purple", alpha=0.3, label="Cum. Reward")
    axes[4].set_title("5. Account Balance & Cumulative Reward", fontsize=14)
    axes[4].set_ylabel("Balance (USD)")
    ax5_reward.set_ylabel("Reward")

    plt.xlabel("Training Steps")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close() # Clean up memory
    print(f"✓ Training performance visualization saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="training_history_high_res.csv")
    parser.add_argument("--output", default="training_performance.png")
    args = parser.parse_args()
    
    visualize_training(args.csv, args.output)
