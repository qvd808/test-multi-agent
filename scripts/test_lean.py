import subprocess

def verify_with_lean(balance: int, cost: int) -> int:
    cmd = ["./proofs/.lake/build/bin/margin_proofs"]
    input_str = f"{balance} {cost}\n"
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate(input=input_str)
    
    if process.returncode != 0:
        raise Exception(f"Lean error: {stderr}")
    
    return int(stdout.strip())

if __name__ == "__main__":
    b, c = 1000, 450
    result = verify_with_lean(b, c)
    print(f"Lean 4 Verified Transition: {b} - {c} = {result}")
    
    if result == (b - c):
        print("✅ Mathematical Verification Passed via Native Lean 4 Process!")
