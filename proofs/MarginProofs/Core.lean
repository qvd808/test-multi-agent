-- MarginProofs/Core.lean (definitions only, no imports needed)
def VETO_PENALTY      : Int := -2000
def HOLDING_SCALE     : Int := 10
def DIRECTION_BONUS   : Int := 50
def INACTION_PENALTY  : Int := -200
def PROFIT_TAKE_BONUS : Int := 1000

def is_valid_trade (balance : Int) (position : Int) (price : Int) (qty : Int) : Bool :=
  if qty > 0 then decide (qty * price ≤ balance)
  else if qty < 0 then decide (0 - qty ≤ position)
  else true

def new_balance (balance : Int) (position : Int) (price : Int) (qty : Int) : Int :=
  if is_valid_trade balance position price qty then balance - (qty * price) else balance

def new_position (balance : Int) (position : Int) (price : Int) (qty : Int) : Int :=
  if is_valid_trade balance position price qty then position + qty else position

def pnl_reward (balance : Int) (position : Int) (price : Int) (qty : Int) (prev_price : Int) : Int :=
  if is_valid_trade balance position price qty then
    (new_balance balance position price qty + new_position balance position price qty * price) - (balance + position * prev_price)
  else 0

def holding_penalty (position : Int) (qty : Int) : Int :=
  -Int.ofNat (Int.natAbs (position + qty)) * HOLDING_SCALE

def direction_bonus (qty : Int) (price : Int) (prev_price : Int) : Int :=
  if qty = 0 then 0
  else
    let price_move := price - prev_price
    let is_correct := (qty > 0 && price_move > 0) || (qty < 0 && price_move < 0)
    let magnitude := if price_move < -50 then 50 else if price_move > 50 then 50 else if price_move < 0 then -price_move else price_move
    let scale := if qty < 0 then -qty else qty
    let base := magnitude * scale
    if is_correct then base else -base

def profit_taking_bonus (position : Int) (qty : Int) (price : Int) (prev_price : Int) : Int :=
  let is_closing := (position > 0 && qty < 0) || (position < 0 && qty > 0)
  if is_closing then
    let closed_qty := if Int.natAbs position < Int.natAbs qty then Int.ofNat (Int.natAbs position) else Int.ofNat (Int.natAbs qty)
    let profit := (price - prev_price) * closed_qty
    if profit > 0 then profit + PROFIT_TAKE_BONUS else profit - 200
  else 0

def inaction_penalty (qty : Int) : Int :=
  if qty = 0 then INACTION_PENALTY else 0

@[export lean_trade_reward]
def c_trade_reward (balance : Int) (position : Int) (price : Int) (qty : Int) (prev_price : Int) : Int :=
  if ¬(is_valid_trade balance position price qty) then VETO_PENALTY
  else
    let unrealized := pnl_reward balance position price qty prev_price
    let holding := holding_penalty position qty
    let direction := direction_bonus qty price prev_price
    let realized := profit_taking_bonus position qty price prev_price
    let inactive := inaction_penalty qty
    unrealized + holding + direction + realized + inactive

@[export lean_trade_balance]
def c_trade_balance (balance : Int) (position : Int) (price : Int) (qty : Int) (_ : Int := 0) : Int :=
  new_balance balance position price qty

@[export lean_trade_position]
def c_trade_position (balance : Int) (position : Int) (price : Int) (qty : Int) (_ : Int := 0) : Int :=
  new_position balance position price qty