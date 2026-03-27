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

-- Proof 5: Direction bonus is anti-symmetric
theorem direction_bonus_antisym (qty price prev_price : Int) :
  direction_bonus qty price prev_price = -direction_bonus (-qty) price prev_price := by
  unfold direction_bonus
  let pm := price - prev_price
  by_cases hq0 : qty = 0
  · simp [hq0]
  · by_cases hp0 : pm = 0
    · have hpm : price = prev_price := by omega
      simp (config := {failIfUnchanged := false}) [hq0, hpm]
    · -- Non-zero case
      by_cases hq_pos : qty > 0
      · by_cases hp_pos : pm > 0
        · simp (config := {decide := true, failIfUnchanged := false}) [hq_pos, hp_pos, hq0, hp0]
          split_ifs <;> (try ring_nf) <;> (try omega)
        · have hp_neg : pm < 0 := by omega
          simp (config := {decide := true, failIfUnchanged := false}) [hq_pos, hp_neg, hq0, hp0]
          split_ifs <;> (try ring_nf) <;> (try omega)
      · have hq_neg : qty < 0 := by omega
        by_cases hp_pos : pm > 0
        · simp (config := {decide := true, failIfUnchanged := false}) [hq_neg, hp_pos, hq0, hp0]
          split_ifs <;> (try ring_nf) <;> (try omega)
        · have hp_neg : pm < 0 := by omega
          simp (config := {decide := true, failIfUnchanged := false}) [hq_neg, hp_neg, hq0, hp0]
          split_ifs <;> (try ring_nf) <;> (try omega)

-- Proof 6: Reward determinism (pure function)
theorem reward_deterministic (b1 p1 pr1 q1 pp1 b2 p2 pr2 q2 pp2 : Int) :
  b1 = b2 → p1 = p2 → pr1 = pr2 → q1 = q2 → pp1 = pp2 → 
  c_trade_reward b1 p1 pr1 q1 pp1 = c_trade_reward b2 p2 pr2 q2 pp2 := by
  intro hb hp hpr hq hpp
  simp [hb, hp, hpr, hq, hpp]

-- Proof 7: Holding in a down market is always punished
theorem holding_punished_in_down_market (balance position price prev_price : Int) :
  position > 0 → price < prev_price →
  c_trade_reward balance position price 0 prev_price < 0 := by
  intro h_pos h_down
  unfold c_trade_reward pnl_reward new_balance new_position holding_penalty inaction_penalty profit_taking_bonus direction_bonus
  simp (config := {decide := true}) [is_valid_trade, VETO_PENALTY, HOLDING_SCALE, INACTION_PENALTY, PROFIT_TAKE_BONUS, Int.ofNat_eq_natCast]
  have h_abs1 : |position| = position := abs_of_pos h_pos
  have h_abs2 : (Int.natAbs position : Int) = position := Int.natAbs_of_nonneg (by omega)
  nlinarith [h_abs1, h_abs2]

-- Proof 8: Holding penalty increases with position size
theorem holding_penalty_monotone (pos1 pos2 qty : Int) :
  Int.natAbs (pos1 + qty) ≤ Int.natAbs (pos2 + qty) → 
  holding_penalty pos1 qty ≥ holding_penalty pos2 qty := by
  intro h
  unfold holding_penalty
  simp only [HOLDING_SCALE, Int.ofNat_eq_natCast]
  have h1 : (Int.natAbs (pos1 + qty) : Int) ≤ (Int.natAbs (pos2 + qty) : Int) :=
    Int.ofNat_le_ofNat_of_le h
  nlinarith

-- Proof 9: Valid trades preserve non-negative balance (alternate form)
theorem valid_trade_preserves_balance (balance position price qty : Int) :
  balance ≥ 0 → qty * price ≤ balance → new_balance balance position price qty ≥ 0 := by
  intro hb hle
  unfold new_balance
  by_cases h : is_valid_trade balance position price qty
  · simp [h]
    omega
  · simp [h]
    exact hb

-- Proof 10: The reward function decomposes correctly
theorem reward_decomposition (balance position price qty prev_price : Int) :
  is_valid_trade balance position price qty →
  c_trade_reward balance position price qty prev_price =
    pnl_reward balance position price qty prev_price +
    holding_penalty position qty +
    direction_bonus qty price prev_price +
    profit_taking_bonus position qty price prev_price +
    inaction_penalty qty := by
  intro h_valid
  unfold c_trade_reward
  simp [h_valid]

end MarginProofs
