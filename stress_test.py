import time
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from env.paper_env import MarginGuardEnv
from datetime import datetime, timedelta

# ── GLOBAL CONFIGURATION ──────────────────────────────────────────────────────
TEST_DURATION_MINS = 30  # Change to 30, 60, or 120 easily
TICKER             = "ETH-USD"
MODEL_PATH         = "margin_guard_pro_v3_ppo"
SAVE_CSV           = True
# ──────────────────────────────────────────────────────────────────────────────

def run_stress_test():
    duration_secs = TEST_DURATION_MINS * 60
    print(f"--- MarginGuard Stress Test Initialized ---")
    print(f"Target Duration : {TEST_DURATION_MINS} minutes")
    print(f"Kernel Type     : Lean 4 Verified FFI")
    
    # Initialize Environment & Model
    env = MarginGuardEnv(ticker=TICKER, initial_balance=50_000, history_length=5, use_cache=True)
    model = PPO.load(MODEL_PATH)
    
    start_time = time.time()
    end_time = start_time + duration_secs
    
    all_history  = []
    total_steps  = 0
    total_vetoes = 0
    episode_count = 0

    try:
        while time.time() < end_time:
            episode_count += 1
            obs, _ = env.reset()
            done = False
            
            while not done and time.time() < end_time:
                # Agent Prediction
                action_idx, _ = model.predict(obs, deterministic=True)
                
                # Verified Step via Lean Kernel
                obs, reward, terminated, truncated, info = env.step(action_idx)
                done = terminated or truncated
                
                total_steps += 1
                is_veto = info.get("vetoed", False)
                if is_veto:
                    total_vetoes += 1

                # Buffer data for post-test analysis
                all_history.append({
                    "step": total_steps,
                    "price": info["price"],
                    "balance": info["balance"],
                    "position": info["position"],
                    "action": env.ACTION_MAP[int(action_idx)],
                    "reward": reward,
                    "vetoed": is_veto
                })

                # Heartbeat every 1000 steps
                if total_steps % 1000 == 0:
                    elapsed = time.time() - start_time
                    fps = total_steps / elapsed
                    remaining = max(0, int(end_time - time.time()))
                    print(f"[{timedelta(seconds=int(elapsed))}] "
                          f"Steps: {total_steps} | "
                          f"FPS: {fps:.1f} | "
                          f"Vetoes: {total_vetoes} | "
                          f"Remaining: {timedelta(seconds=remaining)}")

        print("\n--- Stress Test Complete ---")
        
    except KeyboardInterrupt:
        print("\n--- Test Aborted by User ---")

    # ── Final Analysis & CSV Export ──────────────────────────────────────────
    if SAVE_CSV and all_history:
        df = pd.DataFrame(all_history)
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"margin_guard_audit_{ts}.csv"
        df.to_csv(filename, index=False)
        
        final_bal = df['balance'].iloc[-1]
        v_rate = (total_vetoes / total_steps) * 100
        
        print(f"\nFinal Report:")
        print(f"  - Total Steps    : {total_steps}")
        print(f"  - Episodes       : {episode_count}")
        print(f"  - Final Balance  : ${final_bal:,.2f}")
        print(f"  - Global Veto %  : {v_rate:.2f}%")
        print(f"  - Log Saved to   : {filename}")

if __name__ == "__main__":
    run_stress_test()