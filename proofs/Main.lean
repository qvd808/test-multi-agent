import MarginProofs.Basic

-- Simple CLI for Python Interface
def main : IO Unit := do
  let stdin ← IO.getStdin
  let line ← stdin.getLine
  let parts := line.splitOn " "
  match parts with
  | [balanceStr, costStr] =>
    let balance := balanceStr.trim.toInt!
    let cost := costStr.trim.toInt!
    let result := update_balance balance cost
    IO.println s!"{result}"
  | _ =>
    IO.println "Error: Expected 'balance cost'"
