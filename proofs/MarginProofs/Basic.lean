-- 1. The Core Invariant Logic
-- Evaluates if a trade is mathematically valid.
def is_valid_trade (balance : Int) (position : Int) (price : Int) (qty : Int) : Bool :=
  if qty > 0 then
    -- Buying: Ensure we have enough cash
    (qty * price) <= balance 
  else
    -- Selling: Ensure we have enough shares (qty is negative here, so 0 - qty is positive)
    (0 - qty) <= position    

-- 2. State Transition: Balance
@[export c_trade_balance]
def c_trade_balance (balance : Int) (position : Int) (price : Int) (qty : Int) : Int :=
  if is_valid_trade balance position price qty then
    balance - (qty * price)
  else
    balance -- VETO: Keep original balance

-- 3. State Transition: Position
@[export c_trade_position]
def c_trade_position (balance : Int) (position : Int) (price : Int) (qty : Int) : Int :=
  if is_valid_trade balance position price qty then
    position + qty
  else
    position -- VETO: Keep original position

-- 4. The Shield's Reward System
@[export c_trade_reward]
def c_trade_reward (balance : Int) (position : Int) (price : Int) (qty : Int) : Int :=
  if is_valid_trade balance position price qty then
    0 -- Safe trade. (Python can add profit-based rewards on top of this)
  else
    -1000 -- MASSIVE PENALTY: The agent hit the Lean shield.
