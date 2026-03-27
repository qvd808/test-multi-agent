import MarginProofs.Core
import Mathlib.Tactic
namespace MarginProofs
theorem test (position : Int) (h : position > 0) : True := by
  have := holding_penalty position 0
  set_option pp.all true
  sorry
end MarginProofs
