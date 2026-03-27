# leanffi.py - Optimized Lean FFI for MarginGuard
import ctypes
import os
import subprocess
from pathlib import Path

class LeanBuildError(Exception):
    pass

class MarginGuardCore:
    _SCRIPT_DIR = Path(__file__).parent.resolve()
    _PROOFS_DIR = _SCRIPT_DIR / "proofs"
    _BUILD_DIR = _PROOFS_DIR / ".lake" / "build" / "lib"
    _SO_PATH = _BUILD_DIR / "libmargin_proofs.so"
    _CORE_C = _PROOFS_DIR / ".lake" / "build" / "ir" / "MarginProofs" / "Core.c"
    _WRAPPER_C = _PROOFS_DIR / "c_wrapper" / "margin_proofs.c"
    
    def __init__(self, force_rebuild: bool = False):
        self._lib = None
        self._lean_prefix = Path(self._run(["lean", "--print-prefix"]))
        self._ensure_built(force_rebuild)
        self._load_dependencies()
        self._load_library()
        self._init_lean()
        self._setup_functions()
    
    def _run(self, cmd, cwd=None):
        try:
            return subprocess.check_output(cmd, cwd=cwd or self._PROOFS_DIR, text=True, stderr=subprocess.STDOUT).strip()
        except subprocess.CalledProcessError as e:
            raise LeanBuildError(f"Command failed: {' '.join(cmd)}\nOutput: {e.output}")

    def _ensure_built(self, force):
        should_rebuild = force or not self._SO_PATH.exists()
        if not should_rebuild:
            so_time = self._SO_PATH.stat().st_mtime
            if self._WRAPPER_C.stat().st_mtime > so_time:
                should_rebuild = True
            elif any(f.stat().st_mtime > so_time for f in self._PROOFS_DIR.rglob("*.lean") if not f.is_dir()):
                should_rebuild = True
        
        if should_rebuild:
            self._build()

    def _build(self):
        print("[MarginGuard] Building verified core...")
        self._run(["lake", "build"])
        self._BUILD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Build using leanc
        cmd = [
            "leanc", "-shared", "-fPIC", "-O3",
            "-o", str(self._SO_PATH),
            str(self._CORE_C),
            str(self._WRAPPER_C)
        ]
        self._run(cmd)
        print(f"[MarginGuard] Built: {self._SO_PATH.name}")

    def _load_dependencies(self):
        """Pre-load Lean shared libraries into global namespace to resolve symbols."""
        lean_lib_dir = self._lean_prefix / "lib" / "lean"
        deps = ["libleanshared.so", "libInit_shared.so", "libLean_shared.so"]
        for dep in deps:
            dep_path = lean_lib_dir / dep
            if dep_path.exists():
                try:
                    ctypes.CDLL(str(dep_path), mode=ctypes.RTLD_GLOBAL)
                except Exception as e:
                    print(f"[MarginGuard] Warning: Failed to pre-load {dep}: {e}")

    def _load_library(self):
        try:
            self._lib = ctypes.CDLL(str(self._SO_PATH), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            raise LeanBuildError(f"Failed to load verified core: {e}")

    def _init_lean(self):
        try:
            self._lib.margin_proofs_init()
        except AttributeError:
            raise LeanBuildError("Verified core missing initialization function")

    def _setup_functions(self):
        # We use the cleaned up C wrapper functions
        self._lib.c_trade_reward.argtypes = [ctypes.c_int64] * 7
        self._lib.c_trade_reward.restype = ctypes.c_int64
        
        self._lib.c_trade_balance.argtypes = [ctypes.c_int64] * 4
        self._lib.c_trade_balance.restype = ctypes.c_int64
        
        self._lib.c_trade_position.argtypes = [ctypes.c_int64] * 4
        self._lib.c_trade_position.restype = ctypes.c_int64

    def trade(self, balance, position, price, qty, prev_price, avg_entry, sma_week):
        # All monetary values are processed in cents internally for precision (verified Core.lean)
        # Python handles the dollar <-> cent conversion.
        b_cts = int(round(balance * 100))
        p_val = int(position)
        c_cts = int(round(price * 100))
        q_val = int(qty)
        pp_cts = int(round(prev_price * 100))
        ae_cts = int(round(avg_entry * 100))
        sw_cts = int(round(sma_week * 100))
        
        new_bal = self._lib.c_trade_balance(b_cts, p_val, c_cts, q_val) / 100.0
        new_pos = self._lib.c_trade_position(b_cts, p_val, c_cts, q_val)
        reward_cts = self._lib.c_trade_reward(b_cts, p_val, c_cts, q_val, pp_cts, ae_cts, sw_cts)
        reward = reward_cts / 100.0
        
        return new_bal, new_pos, reward

# Singleton
_core = None
def get_core():
    global _core
    if _core is None: _core = MarginGuardCore()
    return _core

def trade(balance, position, price, qty, prev_price, avg_entry, sma_week):
    return get_core().trade(balance, position, price, qty, prev_price, avg_entry, sma_week)

if __name__ == "__main__":
    print("--- Lean FFI Self Test ---")
    try:
        # Rebuild to ensure everything is fresh
        test_core = MarginGuardCore(force_rebuild=True)
        
        # 1. Test Inaction Penalty
        res = test_core.trade(500.0, 0, 100.0, 0, 100.0, 0, 100.0)
        print(f"Test Hold (Inaction): {res}")
        assert res[2] == -1.8, f"Expected reward -1.8 (INACTION_PENALTY=-180 cents), got {res[2]}"
        
        # 2. Test Success Buy
        res_buy = test_core.trade(1000.0, 0, 100.0, 5, 100.0, 0, 100.0)
        print(f"Test Buy (SUCCESS):   {res_buy}")
        assert res_buy[0] == 500.0, f"Expected balance 500.0, got {res_buy[0]}"
        assert res_buy[1] == 5, f"Expected position 5, got {res_buy[1]}"
        
        # 3. Test Veto (Insufficient Balance)
        res_veto = test_core.trade(100.0, 0, 1000.0, 1, 1000.0, 0, 1000.0)
        print(f"Test Buy (VETO):      {res_veto}")
        # VETO_PENALTY is -5000 cents = -50.0 dollars
        assert res_veto[2] == -50.0, f"Expected veto reward -50.0, got {res_veto[2]}"
        assert res_veto[0] == 100.0, "Balance should be unchanged on veto"
        
        print("\nAll verified FFI tests passed!")
    except Exception as e:
        print(f"FFI test failed: {e}")
        import traceback
        traceback.print_exc()