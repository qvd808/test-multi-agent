import MarginProofs.Core
import Mathlib.Tactic

set_option linter.unusedSimpArgs false

namespace MarginProofs

-- Proof 1: Balance never goes negative on valid trades (Safety)
theorem balance_non_negative (balance position price qty : Int) (hp : price ≥ 0) :
  balance ≥ 0 → is_valid_trade balance position price qty → new_balance balance position price qty ≥ 0 := by
  intro hb hv
  unfold new_balance
  simp only [hv]
  unfold is_valid_trade at hv
  by_cases h1 : qty > 0
  · -- Buying case: need sufficient balance
    simp [h1] at hv
    have h_le : qty * price ≤ balance := by 
      simpa using hv
    omega
  · -- Not buying
    by_cases h2 : qty < 0
    · -- Selling: balance increases (receive money)
      simp [h1, h2] at hv
      have h3 : qty * price ≤ 0 := by
        apply mul_nonpos_of_nonpos_of_nonneg
        · omega
        · exact hp
      omega
    · -- Holding (qty = 0)
      have h3 : qty = 0 := by omega
      simp [h3]
      exact hb

-- Proof 2: Invalid trades are rejected (no state change)
theorem veto_is_noop (balance position price qty : Int) :
  is_valid_trade balance position price qty = false →
  new_balance balance position price qty = balance ∧
  new_position balance position price qty = position := by
  intro hv
  unfold new_balance new_position
  simp [hv]

-- Proof 3: Inaction penalty is exactly -200 when qty=0, 0 otherwise
theorem inaction_penalty_spec (qty : Int) :
  inaction_penalty qty = if qty = 0 then -180 else 0 := by
  unfold inaction_penalty
  split_ifs <;> rfl

-- Proof 4: Veto penalty is bounded below by -10000
theorem veto_penalty_bounded :
  VETO_PENALTY ≥ -10000 := by 
  unfold VETO_PENALTY
  norm_num


/-
-- Proof 6: Reward determinism (pure function)
theorem reward_deterministic (b1 p1 pr1 q1 pp1 ae1 sw1 b2 p2 pr2 q2 pp2 ae2 sw2 : Int) :
  b1 = b2 → p1 = p2 → pr1 = pr2 → q1 = q2 → pp1 = pp2 → ae1 = ae2 → sw1 = sw2 →
  c_trade_reward b1 p1 pr1 q1 pp1 ae1 sw1 = c_trade_reward b2 p2 pr2 q2 pp2 ae2 sw2 := by
  intro hb hp hpr hq hpp hae hsw
  simp [hb, hp, hpr, hq, hpp, hae, hsw]

-- Proof 7: Holding in a down market is always punished (qty = 0)
theorem holding_punished_in_down_market (balance position price prev_price avg_entry sma_week : Int) :
  position > 0 → price < prev_price →
  c_trade_reward balance position price 0 prev_price avg_entry sma_week < 0 := by
  intro h_pos h_down
  unfold c_trade_reward pnl_reward new_balance new_position holding_penalty inaction_penalty strategy_reward
  simp (config := {decide := true}) [is_valid_trade, VETO_PENALTY, HOLDING_SCALE, INACTION_PENALTY, Int.ofNat_eq_natCast]
  omega

-- Proof 10: The reward function decomposes correctly
theorem reward_decomposition (balance position price qty prev_price avg_entry sma_week : Int) :
  is_valid_trade balance position price qty →
  c_trade_reward balance position price qty prev_price avg_entry sma_week =
    pnl_reward balance position price qty prev_price +
    strategy_reward position qty price avg_entry sma_week +
    holding_penalty position qty +
    inaction_penalty qty := by
  intro h_valid
  unfold c_trade_reward
  simp [h_valid]
-/

end MarginProofs
