import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from env.paper_env import MarginGuardEnv
import os

class RewardShapingCallback(BaseCallback):
    """
    Diagnostic callback to monitor vetoes and verified rewards in TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.vetoes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("vetoed", False):
                self.vetoes += 1
            
            # Log metrics to TensorBoard
            self.logger.record("env/reward", info.get("reward", 0))
            self.logger.record("env/balance", info.get("balance", 0))
            self.logger.record("env/position", info.get("position", 0))
            self.logger.record("env/cumulative_vetoes", self.vetoes)
        return True

def train():
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
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # Encourage exploration
        tensorboard_log="./logs/margin_guard_v3/"
    )

    # ── 3. Train ───────────────────────────────────────────────────────────────
    print("\n[agent] Starting training of Verfied PPO (v3)...")
    callback = RewardShapingCallback()
    
    # Train for 200k steps for a "stable" trader
    model.learn(
        total_timesteps=200_000,
        callback=callback,
        progress_bar=True
    )

    # ── 4. Save ────────────────────────────────────────────────────────────────
    model_path = "margin_guard_pro_v3_ppo"
    model.save(model_path)
    print(f"[agent] Training complete. Model saved to {model_path}.zip")

if __name__ == "__main__":
    # Ensure log directories exist
    os.makedirs("./logs/margin_guard_v3/", exist_ok=True)
    train()