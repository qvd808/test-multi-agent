-- 1. Define our function with C-export attribute
@[export update_balance_c]
def update_balance (current_balance : Int) (trade_cost : Int) : Int :=
  current_balance - trade_cost

-- 2. State the theorem
theorem zero_cost_safe (balance : Int) : update_balance balance 0 = balance := by
  simp [update_balance]

theorem margin_comm (margin_a margin_b : Nat) : margin_a + margin_b = margin_b + margin_a := by
  rw [Nat.add_comm]
