-- MarginProofs/Proofs.lean - The proofs (no exports, changes frequently)
import MarginProofs.Core

namespace MarginProofs

theorem balance_non_negative (balance position price qty : Int) (hp : price ≥ 0) :
  balance ≥ 0 → is_valid_trade balance position price qty → new_balance balance position price qty ≥ 0 := by
  intro hb hv
  unfold new_balance
  simp [hv]
  unfold is_valid_trade at hv
  by_cases h1 : qty > 0
  · -- buying case: qty > 0
    simp [h1] at hv
    -- hv is now directly: qty * price ≤ balance
    omega
  · -- qty ≤ 0
    by_cases h2 : qty < 0
    · -- selling case: qty < 0
      simp [h1, h2] at hv
      have h3 : qty * price ≤ 0 := mul_nonpos_of_nonpos_of_nonneg (by omega) hp
      omega
    · -- holding case: qty = 0
      have h3 : qty = 0 := by omega
      simp [h3]
      exact hb

theorem veto_is_noop (balance position price qty : Int) :
  is_valid_trade balance position price qty = false →
  new_balance balance position price qty = balance ∧
  new_position balance position price qty = position := by
  intro hv
  unfold new_balance new_position
  simp [hv]

theorem inaction_penalty_non_positive (qty : Int) :
  (if qty == 0 then INACTION_PENALTY else 0) ≤ 0 := by
  split_ifs <;> decide

end MarginProofs