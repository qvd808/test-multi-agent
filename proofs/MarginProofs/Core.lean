-- MarginProofs/Core.lean

def VETO_PENALTY      : Int := -5000
def HOLDING_SCALE     : Int := 10
def DIRECTION_BONUS   : Int := 50
def INACTION_PENALTY  : Int := -180
def STRATEGY_BONUS    : Int := 3000
def BAD_EXIT_PENALTY  : Int := -1000

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

def inaction_penalty (qty : Int) : Int :=
  if qty = 0 then INACTION_PENALTY else 0

-- Strategy Audit: Proof that Selling at t > Entry results in profit
-- Also considers 1-week SMA for "market phase" context
def strategy_reward (position : Int) (qty : Int) (price : Int) (avg_entry : Int) (sma_week : Int) : Int :=
  let is_closing := (position > 0 && qty < 0) || (position < 0 && qty > 0)
  if is_closing then
    let is_profitable := (position > 0 && price > avg_entry) || (position < 0 && price < avg_entry)
    let trend_bonus := if (position > 0 && price > sma_week) || (position < 0 && price < sma_week) then 500 else 0
    if is_profitable then STRATEGY_BONUS + trend_bonus else BAD_EXIT_PENALTY
  else 0

@[export lean_trade_reward]
def c_trade_reward (balance : Int) (position : Int) (price : Int) (qty : Int) 
                   (prev_price : Int) (avg_entry : Int) (sma_week : Int) : Int :=
  if ¬(is_valid_trade balance position price qty) then VETO_PENALTY
  else
    let pnl      := pnl_reward balance position price qty prev_price
    let strategy := strategy_reward position qty price avg_entry sma_week
    let holding  := holding_penalty position qty
    let inactive := inaction_penalty qty
    pnl + strategy + holding + inactive

@[export lean_trade_balance]
def c_trade_balance (balance : Int) (position : Int) (price : Int) (qty : Int) (_ : Int := 0) : Int :=
  new_balance balance position price qty

@[export lean_trade_position]
def c_trade_position (balance : Int) (position : Int) (price : Int) (qty : Int) (_ : Int := 0) : Int :=
  new_position balance position price qty