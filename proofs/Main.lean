import MarginProofs

-- The Verified Referee CLI — v3
-- Input: "balance position price qty prev_price" (all in cents)
-- Output: "new_balance new_position reward" (all in cents)
def main : IO Unit := do
  let stdin ← IO.getStdin
  let line ← stdin.getLine
  let parts := line.trim.splitOn " "
  match parts with
  | [balanceStr, positionStr, priceStr, qtyStr, prevPriceStr] =>
    let balance    := balanceStr.toInt!
    let position   := positionStr.toInt!
    let price      := priceStr.toInt!
    let qty        := qtyStr.toInt!
    let prev_price := prevPriceStr.toInt!
    
    let nb  := c_trade_balance   balance position price qty prev_price
    let np  := c_trade_position  balance position price qty prev_price
    let rw  := c_trade_reward    balance position price qty prev_price
    
    IO.println s!"{nb} {np} {rw}"
  | _ =>
    IO.eprintln "Error: Expected 'balance position price qty prev_price'"
    IO.Process.exit 1