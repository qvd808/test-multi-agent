import MarginProofs.Basic

-- The Verified Referee CLI
-- Input: "balance position price qty"
-- Output: "new_balance new_position reward"
def main : IO Unit := do
  let stdin ← IO.getStdin
  let line ← stdin.getLine
  let parts := line.splitOn " "
  match parts with
  | [balanceStr, positionStr, priceStr, qtyStr] =>
    let balance := balanceStr.trim.toInt!
    let position := positionStr.trim.toInt!
    let price := priceStr.trim.toInt!
    let qty := qtyStr.trim.toInt!
    
    let nb := c_trade_balance balance position price qty
    let np := c_trade_position balance position price qty
    let rw := c_trade_reward balance position price qty
    
    IO.println s!"{nb} {np} {rw}"
  | _ =>
    IO.println "Error: Expected 'balance position price qty'"
