"""
MarginGuard Visualizer — v4 (Core.lean Audit Edition)
Synchronized with Lean Core.lean verified logic.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO
from env.paper_env import MarginGuardEnv
from datetime import datetime

# ── Sync with Core.lean Constants (in cents) ──────────────────────────────────
VETO_PENALTY      = -5000     # -$50.00
HOLDING_SCALE     = 10        # $0.10 per lot
STRATEGY_BONUS    = 3000      # +$30.00 baseline
BAD_EXIT_PENALTY  = -1000     # -$10.00
INACTION_PENALTY  = -180      # -$1.80


def _decompose_reward(reward_cents, action_qty, pos_before, price, avg_entry, sma_week):
    """
    Mirrors the logic in Core.lean v4 to audit reward components.
    """
    r = reward_cents

    # 1. Veto Audit
    if abs(r - VETO_PENALTY) < 1:
        return {"pnl": 0, "holding": 0, "strategy": 0, 
                "inaction": 0, "veto": VETO_PENALTY, "vetoed": True}

    # 2. Inaction Penalty
    inaction = INACTION_PENALTY if action_qty == 0 else 0

    # 3. Holding Penalty
    new_pos = pos_before + action_qty
    holding = -abs(new_pos) * HOLDING_SCALE

    # 4. Strategy Reward (Smart Exit Proof)
    strategy = 0
    is_closing = (pos_before > 0 and action_qty < 0) or (pos_before < 0 and action_qty > 0)
    if is_closing:
        price_cts = int(round(price * 100))
        entry_cts = int(round(avg_entry * 100))
        sma_cts = int(round(sma_week * 100))
        
        is_profitable = (pos_before > 0 and price_cts > entry_cts) or (pos_before < 0 and price_cts < entry_cts)
        trend_bonus = 500 if ((pos_before > 0 and price_cts > sma_cts) or (pos_before < 0 and price_cts < sma_cts)) else 0
        
        if is_profitable:
            strategy = STRATEGY_BONUS + trend_bonus
        else:
            strategy = BAD_EXIT_PENALTY

    # 5. Raw PnL (Residual)
    pnl = r - holding - strategy - inaction

    return {
        "pnl": pnl, "holding": holding, "strategy": strategy,
        "inaction": inaction, "veto": 0, "vetoed": False
    }


def run_visualization(model_path="margin_guard_pro_v3_ppo", steps=200, save_path="performance_plot.png"):
    print(f"--- MarginGuard Proof-Aware Visualizer ({steps} steps) ---")

    # Ensure your MarginGuardEnv is calling 'lean_trade_reward' in its _consult_lean method!
    env = MarginGuardEnv(ticker="ETH-USD", initial_balance=50_000, history_length=5, use_cache=True)
    
    # Load model
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model {model_path}.zip not found.")
        return
    model = PPO.load(model_path)

    records = []
    obs, _ = env.reset()

    for i in range(steps):
        action_idx, _ = model.predict(obs, deterministic=True)
        action_qty = env.ACTION_MAP[int(action_idx)]

        pos_before = env.position
        bal_before = env.balance
        avg_entry_before = env.avg_entry_price

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action_idx)
        
        # In v4, _build_obs returns (obs, sma_week)
        # But here we need to extract sma_week from the same logic or the latest environment state
        _, sma_week = env._build_obs()

        # Convert Lean reward (float dollars) back to integer cents for decomposition
        reward_cents = int(round(reward * 100))
        
        decomp = _decompose_reward(reward_cents, action_qty, pos_before, info["price"], avg_entry_before, sma_week)

        records.append({
            "step": i,
            "price": info["price"],
            "balance": bal_before,
            "position": pos_before,
            "action": action_qty,
            "reward": reward,
            "portfolio": bal_before + pos_before * info["price"],
            "vetoed": info["vetoed"],
            **decomp
        })
        if terminated or truncated: break

    # --- Unpack Data ---
    T = [r["step"] for r in records]
    prices = [r["price"] for r in records]
    actions = [r["action"] for r in records]
    vetoed = [r["vetoed"] for r in records]
    portfolios = [r["portfolio"] for r in records]
    
    # Components (Converted to USD for plotting)
    pnl_c = [r["pnl"]/100 for r in records]
    hld_c = [r["holding"]/100 for r in records]
    str_c = [r["strategy"]/100 for r in records]
    inc_c = [r["inaction"]/100 for r in records]

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 22))
    gs = gridspec.GridSpec(7, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # P1: Price & Strategy Markers
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(T, prices, color="gray", alpha=0.3, label="Market Price")
    
    buy_steps = [r["step"] for r in records if r["action"] > 0 and not r["vetoed"]]
    buy_prices = [r["price"] for r in records if r["action"] > 0 and not r["vetoed"]]
    sell_steps = [r["step"] for r in records if r["action"] < 0 and not r["vetoed"]]
    sell_prices = [r["price"] for r in records if r["action"] < 0 and not r["vetoed"]]
    veto_steps = [r["step"] for r in records if r["vetoed"]]
    veto_prices = [r["price"] for r in records if r["vetoed"]]

    ax1.scatter(buy_steps, buy_prices, marker="^", color="green", s=100, label="BUY", zorder=5)
    ax1.scatter(sell_steps, sell_prices, marker="v", color="red", s=100, label="SELL", zorder=5)
    ax1.scatter(veto_steps, veto_prices, marker="x", color="black", s=80, label="VETO", zorder=5)
    
    ax1.set_title("P1 - Market Price & Trade Strategy Execution")
    ax1.legend(loc="upper left")
    
    # P2: Reward Decomposition (Audit)
    ax2 = fig.add_subplot(gs[1, :])
    bottom_pos, bottom_neg = np.zeros(len(T)), np.zeros(len(T))
    comps = [(pnl_c, "blue", "Unrealized"), (hld_c, "red", "Hold Cost"), 
             (str_c, "purple", "Strategy Bonus"), 
             (inc_c, "orange", "Inaction")]

    for data, color, label in comps:
        p = [max(v, 0) for v in data]; n = [min(v, 0) for v in data]
        ax2.bar(T, p, bottom=bottom_pos, color=color, alpha=0.7, label=label)
        ax2.bar(T, n, bottom=bottom_neg, color=color, alpha=0.7)
        bottom_pos += np.array(p); bottom_neg += np.array(n)
    ax2.set_title("P2 - Reward Decomposition (Proof-Verified Components)")
    ax2.legend(loc="upper left", ncol=3, fontsize=8)

    # P3: Veto Tracking
    ax3 = fig.add_subplot(gs[2, 0])
    v_indices = [i for i, v in enumerate(vetoed) if v]
    ax3.scatter(v_indices, [1]*len(v_indices), marker="|", color="red", s=100)
    ax3.set_title(f"P3 - Veto Events (Total: {len(v_indices)})")
    ax3.set_yticks([])

    # P4: Inaction Audit
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.bar(T, inc_c, color="orange")
    ax4.axhline(INACTION_PENALTY/100, color="black", ls="--", lw=0.5)
    ax4.set_title("P4 - Inaction Penalty Check (-$2.00)")

    # P5: Cumulative Reward
    ax5 = fig.add_subplot(gs[3, :])
    ax5.plot(T, np.cumsum([r["reward"] for r in records]), color="magenta")
    ax5.set_title("P5 - Cumulative Reward (Agent Goal)")

    # P6: Portfolio Value
    ax6 = fig.add_subplot(gs[4, :])
    ax6.plot(T, portfolios, color="green")
    ax6.set_title("P6 - Total Portfolio Value (USD)")

    # P7: Action Dist
    ax7 = fig.add_subplot(gs[5, :])
    ax7.hist(actions, bins=21, color="teal", edgecolor="black")
    ax7.set_title("P7 - Action Distribution (Lot Sizes)")

    # Summary Text
    final_p = portfolios[-1]
    ret = (final_p - 50000)/500
    summary = f"Steps: {len(T)} | Portfolio: USD {final_p:,.2f} ({ret:+.1f}%) | Vetoes: {len(v_indices)}"
    fig.text(0.5, 0.02, summary, ha="center", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved visualizer to {save_path}")

if __name__ == "__main__":
    run_visualization(steps=20000)