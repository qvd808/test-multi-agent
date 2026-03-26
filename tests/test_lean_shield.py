import subprocess
import os
import unittest

class TestLeanShield(unittest.TestCase):
    """
    Rigorously tests the Lean 4 'Referee' logic via the Python Pipe interface.
    No network required.
    """
    
    BINARY_PATH = "./proofs/.lake/build/bin/margin_proofs"

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.BINARY_PATH):
            raise FileNotFoundError(f"Lean binary not found at {cls.BINARY_PATH}. Run 'cd proofs && lake build'.")

    def _consult_lean(self, b, p, price, q):
        input_str = f"{int(b)} {int(p)} {int(price)} {int(q)}\n"
        process = subprocess.Popen(
            [self.BINARY_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_str)
        if process.returncode != 0:
            return None, stderr
        
        parts = stdout.strip().split(" ")
        return [int(x) for x in parts], None

    def test_legal_purchase(self):
        # Case: Buy 10 shares at $10. Budget $1000.
        # Cost: 100. New Balance: 900.
        res, err = self._consult_lean(1000, 0, 10, 10)
        self.assertIsNone(err)
        self.assertEqual(res[0], 900)  # New Balance
        self.assertEqual(res[1], 10)   # New Position
        self.assertEqual(res[2], 0)    # Reward (Safe)

    def test_insufficient_funds(self):
        # Case: Try to buy 100 shares at $10. Budget $50.
        # Required: 1000. Balance: 50.
        # Result: VETO (State unchanged, -1000 penalty)
        res, err = self._consult_lean(50, 0, 10, 100)
        self.assertIsNone(err)
        self.assertEqual(res[0], 50)     # Balance unchanged
        self.assertEqual(res[1], 0)      # Position unchanged
        self.assertEqual(res[2], -1000)  # Penalty applied

    def test_legal_sell(self):
        # Case: Sell 5 shares at $20. Have 10 shares.
        # Credit: 100. New Balance: 100 + original.
        res, err = self._consult_lean(1000, 10, 20, -5)
        self.assertIsNone(err)
        self.assertEqual(res[0], 1100)  # New Balance
        self.assertEqual(res[1], 5)     # New Position
        self.assertEqual(res[2], 0)     # Reward (Safe)

    def test_naked_short_protection(self):
        # Case: Try to sell 50 shares. Have 10 shares.
        # Result: VETO (State unchanged, -1000 penalty)
        res, err = self._consult_lean(1000, 10, 10, -50)
        self.assertIsNone(err)
        self.assertEqual(res[0], 1000)   # Balance unchanged
        self.assertEqual(res[1], 10)     # Position unchanged
        self.assertEqual(res[2], -1000)  # Penalty applied

if __name__ == "__main__":
    unittest.main()
