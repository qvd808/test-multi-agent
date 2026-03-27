import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from env.paper_env import MarginGuardEnv
import os

class RewardShapingCallback(BaseCallback):
    """
    Diagnostic callback to monitor vetoes and verified rewards in TensorBoard.
    """
    def __init__(self, verbose=0, csv_path="training_history_high_res.csv"):
        super().__init__(verbose)
        self.vetoes = 0
        self.csv_path = csv_path
        self.history = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("vetoed", False):
                self.vetoes += 1
            
            # Log metrics to TensorBoard
            self.logger.record("env/reward", info.get("reward", 0))
            self.logger.record("env/balance", info.get("balance", 0))
            self.logger.record("env/position", info.get("position", 0))
            self.logger.record("env/price", info.get("price", 0))
            self.logger.record("env/action", info.get("action", 0))
            self.logger.record("env/is_vetoed", 1 if info.get("vetoed", False) else 0)
            self.logger.record("env/cumulative_vetoes", self.vetoes)
            
            # High-resolution history for visualization
            self.history.append({
                "step": self.num_timesteps,
                "price": info.get("price", 0),
                "balance": info.get("balance", 0),
                "position": info.get("position", 0),
                "action": info.get("action", 0),
                "reward": info.get("reward", 0),
                "vetoed": info.get("vetoed", False)
            })

        # Save heartbeat to CSV every 10,000 steps to prevent data loss
        if self.num_timesteps % 10000 == 0 and self.history:
            self._save_to_csv()
            
        return True

    def _on_training_end(self) -> None:
        """Save final history at the end of training."""
        self._save_to_csv()

    def _save_to_csv(self):
        import pandas as pd
        df = pd.DataFrame(self.history)
        # Append if file exists, else write header
        mode = 'a' if os.path.exists(self.csv_path) else 'w'
        header = not os.path.exists(self.csv_path)
        df.to_csv(self.csv_path, mode=mode, header=header, index=False)
        self.history = [] # Clear memory after saving

def train():
    # ── 0. Prep Logging ────────────────────────────────────────────────────────
    csv_path = "training_history_high_res.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"[agent] Cleared existing log: {csv_path}")

    # ── 1. Create Environment ──────────────────────────────────────────────────
    # history_length=5 gives the agent "memory" of recent price trends.
    env = MarginGuardEnv(
        ticker="ETH-USD",
        initial_balance=50_000,
        history_length=5,
        use_cache=True
    )

    # ── 2. Configure PPO Agent ─────────────────────────────────────────────────
    # We use a Multi-Layer Perceptron (MlpPolicy) with normalized inputs.
    model = PPO(
        "MlpPolicy",
        env,
        device="cuda",  # Forces training on the GPU
        verbose=1,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,         # Encourage exploration to find the "Strategy Bonus"
        tensorboard_log="./logs/margin_guard_v4/"
    )

    # ── 3. Train ───────────────────────────────────────────────────────────────
    print("\n[agent] Starting training of Verfied PPO (v3)...")
    callback = RewardShapingCallback(csv_path=csv_path)
    
    # Train for 200k steps for a "stable" trader
    model.learn(
        total_timesteps=300_000,
        callback=callback,
        progress_bar=True
    )

    # ── 4. Save ────────────────────────────────────────────────────────────────
    model_path = "margin_guard_pro_v3_ppo"
    model.save(model_path)
    print(f"[agent] Training complete. Model saved to {model_path}.zip")

if __name__ == "__main__":
    # Ensure log directories exist
    os.makedirs("./logs/margin_guard_v4/", exist_ok=True)
    train()