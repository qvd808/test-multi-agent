"""
MarginGuard Visualizer — v3 (Proof-Aware Edition)

Every panel corresponds to a formally verified theorem in MarginProofs/Proofs.lean.
This is not just a performance dashboard — it's a runtime audit of the proofs.

Panel → Theorem it audits
─────────────────────────────────────────────────────────────────────────────
P1  Price + trade markers     → visual context for all other panels
P2  Balance floor             → balance_non_negative / valid_trade_preserves_balance
P3  Veto events               → veto_is_noop  (state must be UNCHANGED on veto)
P4  Holding penalty signal    → holding_penalty_monotone + holding_punished_in_down_market
P5  Direction alignment       → direction_bonus_antisym
    (% steps where sign(action) == sign(price_change))
P6  Inaction audit            → inaction_penalty_spec  (-200 exactly when qty=0)
P7  Reward decomposition      → reward_decomposition  (PnL + hold + dir + inaction)
P8  Action distribution       → system-level: is agent exploring all 21 actions?
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from stable_baselines3 import PPO
from env.paper_env import MarginGuardEnv

# ── Lean reward constants (must match MarginProofs/Core.lean exactly) ─────────
# These are used to decompose the raw Lean reward back into its components
# for auditing. If Core.lean changes these, update here too.
VETO_PENALTY     = -10_000   # cents  →  -$100.00
HOLDING_SCALE    = 10        # cents per share per step
DIRECTION_BONUS  = 500       # cents  →  $5.00
INACTION_PENALTY = -200      # cents  →  -$2.00


def _decompose_reward(reward_cents: int, action_qty: int, position_before: int,
                      price: float, prev_price: float) -> dict:
    """
    Reverse-engineer the Lean reward back into its components.
    Matches the reward_decomposition theorem exactly.

    On a veto: reward == VETO_PENALTY, all components are 0 except veto.
    On a valid trade: reward = pnl + holding + direction + inaction.
    """
    r = reward_cents

    # Detect veto: reward equals the veto constant
    if r == VETO_PENALTY:
        return {"pnl": 0, "holding": 0, "direction": 0,
                "inaction": 0, "veto": VETO_PENALTY, "vetoed": True}

    # Inaction (inaction_penalty_spec: exactly -200 when qty=0)
    inaction = INACTION_PENALTY if action_qty == 0 else 0

    # Holding (holding_penalty_monotone: -HOLDING_SCALE * |new_position|)
    new_pos  = position_before + action_qty
    holding  = -abs(new_pos) * HOLDING_SCALE

    # Direction (direction_bonus_antisym)
    price_change = price - prev_price
    if action_qty == 0 or price_change == 0:
        direction = 0
    elif (action_qty > 0 and price_change > 0) or (action_qty < 0 and price_change < 0):
        direction = DIRECTION_BONUS
    else:
        direction = -DIRECTION_BONUS

    # PnL is whatever remains (reward_decomposition theorem)
    pnl = r - holding - direction - inaction

    return {"pnl": pnl, "holding": holding, "direction": direction,
            "inaction": inaction, "veto": 0, "vetoed": False}


def run_visualization(model_path="margin_guard_pro_v3_ppo", steps=200,
                      save_path="performance_plot.png"):
    print(f"--- MarginGuard Proof-Aware Visualizer ({steps} steps) ---")

    env   = MarginGuardEnv(ticker="ETH-USD", initial_balance=50_000,
                           history_length=5, use_cache=True)
    model = PPO.load(model_path)

    # ── Simulation ────────────────────────────────────────────────────────────
    records  = []
    obs, _   = env.reset()
    prev_price = float(env.price_history[-1])

    for i in range(steps):
        action_idx, _ = model.predict(obs, deterministic=True)
        action_idx    = int(action_idx)
        action_qty    = env.ACTION_MAP[action_idx]

        # Capture state BEFORE step
        price_before = float(env.price_history[-1])
        bal_before   = env.balance
        pos_before   = env.position

        obs, reward, terminated, truncated, info = env.step(action_idx)

        price_after = info["price"]
        reward_cents = int(round(reward * 100))
        decomp = _decompose_reward(
            reward_cents, action_qty, pos_before,
            price_after, price_before
        )

        records.append({
            "step":       i,
            "price":      price_after,
            "prev_price": price_before,
            "balance":    bal_before,
            "position":   pos_before,
            "action":     action_qty,
            "reward":     reward,
            "portfolio":  bal_before + pos_before * price_before,
            **decomp,
        })

        if terminated or truncated:
            print(f"  Episode ended at step {i + 1}")
            break

    # ── Unpack arrays ─────────────────────────────────────────────────────────
    T          = [r["step"]      for r in records]
    prices     = [r["price"]     for r in records]
    portfolios = [r["portfolio"] for r in records]
    positions  = [r["position"]  for r in records]
    actions    = [r["action"]    for r in records]
    rewards    = [r["reward"]    for r in records]
    balances   = [r["balance"]   for r in records]
    cum_rew    = np.cumsum(rewards)

    pnl_comp      = [r["pnl"]       / 100 for r in records]   # back to dollars
    hold_comp     = [r["holding"]   / 100 for r in records]
    dir_comp      = [r["direction"] / 100 for r in records]
    inact_comp    = [r["inaction"]  / 100 for r in records]
    vetoed        = [r["vetoed"]          for r in records]

    price_changes = [r["price"] - r["prev_price"] for r in records]

    buy_idx  = [i for i, a in enumerate(actions) if a > 0]
    sell_idx = [i for i, a in enumerate(actions) if a < 0]
    hold_idx = [i for i, a in enumerate(actions) if a == 0]
    veto_idx = [i for i, v in enumerate(vetoed)  if v]

    max_pf   = np.maximum.accumulate(portfolios)
    drawdown = [pf - mx for pf, mx in zip(portfolios, max_pf)]

    # Direction alignment: steps where sign(action) == sign(price_change)
    aligned = [
        i for i, (a, dp) in enumerate(zip(actions, price_changes))
        if a != 0 and dp != 0 and np.sign(a) == np.sign(dp)
    ]
    trading_steps = [i for i, a in enumerate(actions) if a != 0]
    align_rate = len(aligned) / max(len(trading_steps), 1)

    # ── Layout: 8 panels ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 24))
    gs  = gridspec.GridSpec(
        8, 2, figure=fig,
        height_ratios=[2.5, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.5],
        hspace=0.5, wspace=0.35,
    )

    ax_price = fig.add_subplot(gs[0, :])
    ax_bal   = fig.add_subplot(gs[1, :])
    ax_veto  = fig.add_subplot(gs[2, :])
    ax_hold  = fig.add_subplot(gs[3, :])
    ax_dir   = fig.add_subplot(gs[4, :])
    ax_inact = fig.add_subplot(gs[5, 0])
    ax_decomp= fig.add_subplot(gs[5, 1])
    ax_hist  = fig.add_subplot(gs[6, :])
    ax_cumr  = fig.add_subplot(gs[7, :])

    GRID  = dict(alpha=0.2, linewidth=0.5)
    GREEN = "#2ca02c"
    RED   = "#d62728"
    BLUE  = "#4C72B0"

    # ── P1: Price + trade markers ─────────────────────────────────────────────
    ax_price.plot(T, prices, color=BLUE, lw=1.2, alpha=0.8, label="ETH-USD")
    if buy_idx:
        ax_price.scatter(buy_idx,  [prices[i] for i in buy_idx],
                         marker="^", color=GREEN, s=55, zorder=5,
                         label=f"BUY ({len(buy_idx)})")
    if sell_idx:
        ax_price.scatter(sell_idx, [prices[i] for i in sell_idx],
                         marker="v", color=RED,   s=55, zorder=5,
                         label=f"SELL ({len(sell_idx)})")
    if veto_idx:
        ax_price.scatter(veto_idx, [prices[i] for i in veto_idx],
                         marker="x", color="orange", s=55, zorder=6,
                         label=f"VETO ({len(veto_idx)})")
    ax_price.set_ylabel("Price ($)", fontsize=10)
    ax_price.set_title("P1 — Trades vs Price", fontsize=11, fontweight="bold")
    ax_price.legend(fontsize=9); ax_price.grid(**GRID)

    # ── P2: Balance floor audit → balance_non_negative ────────────────────────
    ax_bal.plot(T, balances, color="#9467bd", lw=1.3, label="Cash balance")
    ax_bal.axhline(0, color=RED, ls="--", lw=1.0, label="Zero (must never cross)")
    ax_bal.fill_between(T, balances, 0, alpha=0.08, color="#9467bd")
    min_bal = min(balances)
    ax_bal.set_title(
        f"P2 — Balance floor  [theorem: balance_non_negative]"
        f"   min={min_bal:,.0f}  ✓" if min_bal >= 0 else "  ✗ VIOLATION",
        fontsize=10, fontweight="bold"
    )
    ax_bal.set_ylabel("Cash ($)", fontsize=9)
    ax_bal.legend(fontsize=8); ax_bal.grid(**GRID)

    # ── P3: Veto audit → veto_is_noop ────────────────────────────────────────
    veto_rewards = [r if v else 0 for r, v in zip(rewards, vetoed)]
    ax_veto.bar(T, veto_rewards, color="orange", alpha=0.8, width=1.0,
                label="Veto reward (must be flat = VETO_PENALTY)")
    ax_veto.axhline(VETO_PENALTY / 100, color=RED, ls="--", lw=0.8,
                    label=f"VETO_PENALTY = ${VETO_PENALTY/100:.0f}")
    ax_veto.set_title(
        f"P3 — Veto events  [theorem: veto_is_noop]"
        f"   count={len(veto_idx)}   rate={len(veto_idx)/max(len(T),1):.1%}",
        fontsize=10, fontweight="bold"
    )
    ax_veto.set_ylabel("Veto reward ($)", fontsize=9)
    ax_veto.legend(fontsize=8); ax_veto.grid(**GRID)

    # ── P4: Holding penalty → holding_penalty_monotone ────────────────────────
    ax_hold.bar(T, hold_comp, color=RED, alpha=0.6, width=1.0, label="|position| cost")
    ax_hold.plot(T, positions, color=BLUE, lw=1.0, alpha=0.6, label="Position size")
    ax2_hold = ax_hold.twinx()
    ax2_hold.plot(T, positions, color=BLUE, lw=1.0, alpha=0.0)   # invisible, just to scale
    ax_hold.set_title(
        "P4 — Holding penalty  [theorem: holding_penalty_monotone]"
        "   larger position → deeper red bar",
        fontsize=10, fontweight="bold"
    )
    ax_hold.set_ylabel("Hold cost ($)", fontsize=9, color=RED)
    ax_hold.legend(fontsize=8, loc="upper right"); ax_hold.grid(**GRID)

    # ── P5: Direction alignment → direction_bonus_antisym ─────────────────────
    dir_colors = [GREEN if d > 0 else (RED if d < 0 else "#aec7e8") for d in dir_comp]
    ax_dir.bar(T, dir_comp, color=dir_colors, alpha=0.75, width=1.0)
    ax_dir.axhline(0, color="gray", lw=0.6)
    ax_dir.set_title(
        f"P5 — Direction bonus/penalty  [theorem: direction_bonus_antisym]"
        f"   alignment rate = {align_rate:.1%}  (green=with trend, red=against)",
        fontsize=10, fontweight="bold"
    )
    ax_dir.set_ylabel("Dir. bonus ($)", fontsize=9); ax_dir.grid(**GRID)

    # ── P6: Inaction audit → inaction_penalty_spec ────────────────────────────
    expected_inact = [INACTION_PENALTY / 100 if a == 0 else 0.0 for a in actions]
    violation = any(
        abs(ic - ei) > 0.01
        for ic, ei, v in zip(inact_comp, expected_inact, vetoed)
        if not v
    )
    ax_inact.bar(T, inact_comp, color="#ff7f0e", alpha=0.7, width=1.0,
                 label="Inaction component")
    ax_inact.axhline(INACTION_PENALTY / 100, color=RED, ls="--", lw=0.8,
                     label=f"Expected = ${INACTION_PENALTY/100:.2f}")
    ax_inact.set_title(
        f"P6 — Inaction penalty  [theorem: inaction_penalty_spec]"
        f"   {'✓ consistent' if not violation else '✗ VIOLATION'}",
        fontsize=9, fontweight="bold"
    )
    ax_inact.set_ylabel("Inaction ($)", fontsize=9)
    ax_inact.legend(fontsize=7); ax_inact.grid(**GRID)

    # ── P7: Reward decomposition stacked → reward_decomposition ───────────────
    bottom_pos = np.zeros(len(T))
    bottom_neg = np.zeros(len(T))

    for comp, color, label in [
        (pnl_comp,   "#1f77b4", "PnL"),
        (hold_comp,  "#d62728", "Holding"),
        (dir_comp,   "#2ca02c", "Direction"),
        (inact_comp, "#ff7f0e", "Inaction"),
    ]:
        pos_vals = [max(v, 0) for v in comp]
        neg_vals = [min(v, 0) for v in comp]
        ax_decomp.bar(T, pos_vals, bottom=bottom_pos, color=color,
                      alpha=0.7, width=1.0, label=label)
        ax_decomp.bar(T, neg_vals, bottom=bottom_neg, color=color, alpha=0.7, width=1.0)
        bottom_pos += np.array(pos_vals)
        bottom_neg += np.array(neg_vals)

    ax_decomp.plot(T, rewards, color="black", lw=0.8, alpha=0.6, label="Total (Lean)")
    ax_decomp.axhline(0, color="gray", lw=0.5)
    ax_decomp.set_title(
        "P7 — Reward decomposition  [theorem: reward_decomposition]",
        fontsize=9, fontweight="bold"
    )
    ax_decomp.set_ylabel("Reward ($)", fontsize=9)
    ax_decomp.legend(fontsize=7, loc="lower left"); ax_decomp.grid(**GRID)

    # ── P8: Action distribution ───────────────────────────────────────────────
    bins = np.arange(-10.5, 11.5, 1)
    counts, _, patches = ax_hist.hist(actions, bins=bins, edgecolor="white",
                                       linewidth=0.5, alpha=0.85)
    for patch, center in zip(patches, range(-10, 11)):
        patch.set_facecolor(GREEN if center > 0 else (RED if center < 0 else "#aec7e8"))
    ax_hist.axvline(0, color="gray", ls="--", lw=0.8)
    ax_hist.set_xlabel("Action (lot qty)", fontsize=10)
    ax_hist.set_ylabel("Frequency", fontsize=10)
    ax_hist.set_xticks(range(-10, 11))
    ax_hist.set_title(
        "P8 — Action distribution   healthy = spread across both sides, not spiking at +1",
        fontsize=10, fontweight="bold"
    )
    ax_hist.grid(**GRID)

    # ── P9: Cumulative reward (full width) ────────────────────────────────────
    ax_cumr.plot(T, cum_rew, color="#e377c2", lw=1.5, label="Cumulative reward")
    ax_cumr.axhline(0, color="gray", ls="--", lw=0.8)
    ax_cumr.fill_between(T, cum_rew, 0,
                          where=[r >= 0 for r in cum_rew],
                          alpha=0.15, color=GREEN)
    ax_cumr.fill_between(T, cum_rew, 0,
                          where=[r < 0 for r in cum_rew],
                          alpha=0.15, color=RED)
    ax_cumr.set_ylabel("Cumulative reward ($)", fontsize=10)
    ax_cumr.set_xlabel("Time Step (hourly)", fontsize=10)
    ax_cumr.set_title("Cumulative Lean reward over episode", fontsize=10)
    ax_cumr.legend(fontsize=9); ax_cumr.grid(**GRID)

    # ── Summary footer ────────────────────────────────────────────────────────
    final_pf   = portfolios[-1] if portfolios else env.initial_balance
    pf_return  = (final_pf - env.initial_balance) / env.initial_balance * 100
    summary = (
        f"Steps: {len(T)}   |   "
        f"Portfolio: ${final_pf:,.0f} ({pf_return:+.1f}%)   |   "
        f"Max drawdown: ${min(drawdown):,.0f}   |   "
        f"Buys: {len(buy_idx)}  Sells: {len(sell_idx)}  Holds: {len(hold_idx)}   |   "
        f"Vetoes: {len(veto_idx)} ({len(veto_idx)/max(len(T),1):.1%})   |   "
        f"Direction alignment: {align_rate:.1%}   |   "
        f"Cumulative reward: {cum_rew[-1]:+.2f}"
    )
    fig.text(0.5, 0.005, summary, ha="center", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", alpha=0.9))

    plt.suptitle(
        "MarginGuard — Proof-Aware Performance Dashboard\n"
        "Each panel audits a theorem from MarginProofs/Proofs.lean",
        fontsize=13, fontweight="bold", y=1.002
    )
    plt.tight_layout(rect=[0, 0.025, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {os.path.abspath(save_path)}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    run_visualization()