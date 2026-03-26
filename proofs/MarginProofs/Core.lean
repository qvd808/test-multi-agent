import Mathlib

namespace MarginProofs

def VETO_PENALTY     : Int := -10000
def HOLDING_SCALE    : Int := 10
def DIRECTION_BONUS  : Int := 500
def INACTION_PENALTY : Int := -200

def is_valid_trade (balance : Int) (position : Int) (price : Int) (qty : Int) : Bool :=
  if qty > 0 then decide (qty * price ≤ balance)
  else if qty < 0 then decide (0 - qty ≤ position)
  else true

def new_balance (balance : Int) (position : Int) (price : Int) (qty : Int) : Int :=
  if is_valid_trade balance position price qty then balance - (qty * price) else balance

def new_position (balance : Int) (position : Int) (price : Int) (qty : Int) : Int :=
  if is_valid_trade balance position price qty then position + qty else position

@[export c_trade_balance]
def c_trade_balance (balance : Int) (position : Int) (price : Int) (qty : Int) (_ : Int := 0) : Int :=
  new_balance balance position price qty

@[export c_trade_position]
def c_trade_position (balance : Int) (position : Int) (price : Int) (qty : Int) (_ : Int := 0) : Int :=
  new_position balance position price qty

@[export c_trade_reward]
def c_trade_reward (balance : Int) (position : Int) (price : Int) (qty : Int) (prev_price : Int) : Int :=
  if ¬(is_valid_trade balance position price qty) then VETO_PENALTY
  else
    let nb := new_balance balance position price qty
    let np := new_position balance position price qty
    let pnl := (nb + np * price) - (balance + position * prev_price)
    let holding := -(Int.ofNat np.natAbs) * HOLDING_SCALE
    let direction := if (qty > 0 && price - prev_price > 0) || (qty < 0 && price - prev_price < 0) then DIRECTION_BONUS else -DIRECTION_BONUS
    let inaction := if qty == 0 then INACTION_PENALTY else 0
    pnl + holding + direction + inaction

end MarginProofs
