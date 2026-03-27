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
VETO_PENALTY      = -2000     # -$20.00
HOLDING_SCALE     = 10        # $0.10 per lot
DIRECTION_BONUS   = 50        # Base magnitude scale
INACTION_PENALTY  = -200      # -$2.00
PROFIT_TAKE_BONUS = 1000      # +$10.00 incentive


def _decompose_reward(reward_cents, action_qty, pos_before, price, prev_price):
    """
    Mirrors the logic in Core.lean to audit reward components.
    """
    r = reward_cents
    price_move_cents = int(round((price - prev_price) * 100))

    # 1. Veto Audit
    if abs(r - VETO_PENALTY) < 1:
        return {"pnl": 0, "holding": 0, "direction": 0, "realized": 0,
                "inaction": 0, "veto": VETO_PENALTY, "vetoed": True}

    # 2. Inaction Penalty (Theorem: inaction_penalty_spec)
    inaction = INACTION_PENALTY if action_qty == 0 else 0

    # 3. Holding Penalty (Theorem: holding_penalty_monotone)
    new_pos = pos_before + action_qty
    holding = -abs(new_pos) * HOLDING_SCALE

    # 4. Direction Bonus (Dynamic Scaling)
    direction = 0
    if action_qty != 0:
        is_correct = (action_qty > 0 and price_move_cents > 0) or (action_qty < 0 and price_move_cents < 0)
        # Lean magnitude logic: magnitude = clamp(abs(price_move), 0, 50)
        magnitude = min(abs(price_move_cents), 50)
        scale = abs(action_qty)
        base = magnitude * scale
        direction = base if is_correct else -base

    # 5. Profit Taking Bonus (Realized)
    realized = 0
    is_closing = (pos_before > 0 and action_qty < 0) or (pos_before < 0 and action_qty > 0)
    if is_closing:
        closed_qty = min(abs(pos_before), abs(action_qty))
        profit = price_move_cents * closed_qty
        if profit > 0:
            realized = profit + PROFIT_TAKE_BONUS
        else:
            realized = profit - 200

    # 6. Unrealized PnL (Residual)
    pnl = r - holding - direction - realized - inaction

    return {
        "pnl": pnl, "holding": holding, "direction": direction,
        "realized": realized, "inaction": inaction, "veto": 0, "vetoed": False
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

        price_before = float(env.price_history[-1])
        pos_before = env.position
        bal_before = env.balance

        obs, reward, terminated, truncated, info = env.step(action_idx)

        # Convert Lean reward (float dollars) back to integer cents for decomposition
        reward_cents = int(round(reward * 100))
        
        decomp = _decompose_reward(reward_cents, action_qty, pos_before, info["price"], price_before)

        records.append({
            "step": i,
            "price": info["price"],
            "balance": bal_before,
            "position": pos_before,
            "action": action_qty,
            "reward": reward,
            "portfolio": bal_before + pos_before * price_before,
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
    dir_c = [r["direction"]/100 for r in records]
    rel_c = [r["realized"]/100 for r in records]
    inc_c = [r["inaction"]/100 for r in records]

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 22))
    gs = gridspec.GridSpec(7, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # P1: Price
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(T, prices, color="gray", alpha=0.5)
    ax1.set_title("P1 - Market Price & Trade Execution")
    
    # P2: Reward Decomposition (Audit)
    ax2 = fig.add_subplot(gs[1, :])
    bottom_pos, bottom_neg = np.zeros(len(T)), np.zeros(len(T))
    comps = [(pnl_c, "blue", "Unrealized"), (hld_c, "red", "Hold Cost"), 
             (dir_c, "green", "Dir Bonus"), (rel_c, "purple", "Realized Profit"), 
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
    run_visualization()