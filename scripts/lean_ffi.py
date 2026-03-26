import ctypes
import os

# --- 1. How to compile the Lean code into a Shared Library (.so) ---
# To make Lean logic usable in Python, we compile the exported C into a library.
#
# Command (example):
# cc -shared -o MarginProofs.so \
#    -I /root/.elan/toolchains/stable/include \
#    -L /root/.elan/toolchains/stable/lib/lean \
#    .lake/build/ir/MarginProofs/Basic.c \
#    -lleancpp -lutil

class LeanVerifiedCore:
    def __init__(self, lib_path):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Verified library not found at {lib_path}. Run 'lake build' first.")
        
        self.lib = ctypes.CDLL(lib_path)
        
        # 2. Setup Lean Runtime Init
        # Every Lean module has an initialize function
        self.lib.initialize_margin__proofs_MarginProofs_Basic.argtypes = [ctypes.c_uint8]
        self.lib.initialize_margin__proofs_MarginProofs_Basic.restype = ctypes.c_void_p
        
        # Init runtime
        self.lib.initialize_margin__proofs_MarginProofs_Basic(0)
        
        # 3. Define the exported function
        # lean_object* update_balance_c(lean_object*, lean_object*)
        self.lib.update_balance_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.update_balance_c.restype = ctypes.c_void_p

    def update_balance(self, balance: int, cost: int) -> int:
        # Lean uses 'boxed' objects (lean_object*). 
        # For small integers, it uses tagged pointers.
        # lean_box(n) = (n << 1) | 1
        # lean_unbox(obj) = obj >> 1
        
        def box(n): return (n << 1) | 1
        def unbox(obj): return obj >> 1
        
        res_ptr = self.lib.update_balance_c(box(balance), box(cost))
        return unbox(res_ptr)

if __name__ == "__main__":
    print("--- Lean 4 Python Interface ---")
    # This assumes we have compiled the .so
    try:
        core = LeanVerifiedCore("./MarginProofs.so")
        result = core.update_balance(100, 20)
        print(f"Verified Result: {result}")
    except Exception as e:
        print(f"Workflow Note: To run this FFI, you must first link the .c files into a .so.")
        print(f"Current Error (expected): {e}")
