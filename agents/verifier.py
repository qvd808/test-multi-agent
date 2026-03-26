import os
import sys
import random
import numpy as np
import tempfile
import subprocess
from langchain_ollama import ChatOllama
from state import AgentState

def get_llm():
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
    gemini_key = os.environ.get("GOOGLE_API_KEY")

    llm_ollama = ChatOllama(model=model_name, base_url=ollama_url, temperature=0.1)
    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, max_retries=0).with_fallbacks([llm_ollama])
    return llm_ollama

def verifier_node(state: AgentState) -> AgentState:
    print("[verifier] Starting specification-driven verification")

    generated_code = state.get("generated_code", "")
    if not generated_code:
        return {**state, "proof_passed": False, "proof_error": "No code to verify"}

    # 1. Compile the User's Proof to ensure it's mathematically sound
    spec_path = "proofs/trading_agent_proof.v"
    if not os.path.exists(spec_path):
        return {**state, "proof_passed": False, "proof_error": f"Specification not found at {spec_path}"}

    print(f"[verifier] Compiling Coq specification: {spec_path}")
    comp_result = subprocess.run(["coqc", spec_path], capture_output=True, text=True)
    if comp_result.returncode != 0:
        error = comp_result.stderr or comp_result.stdout
        print(f"[verifier] ✗ Specification failed to compile:\n{error[:500]}")
        return {**state, "proof_passed": False, "proof_error": f"Coq Spec Error:\n{error}"}

    # 2. Translate Coq logic to Python Reference (via LLM)
    # This acts as our executable specification.
    with open(spec_path, "r") as f:
        coq_spec = f.read()

    print("[verifier] Translating Coq specification to Python reference...")
    llm = get_llm()
    translation_prompt = f"""You are a formal methods expert. 
Translate the following Coq `step` function and types into a pure Python function `reference_step(params, state, price, action)`.

COQ SPECIFICATION:
{coq_spec}

REQUIREMENTS:
1. Output ONLY the Python code. No explanations.
2. The `state` is a dict: {{'cash': float, 'shares_held': int}}.
3. The `params` is a dict: {{'commission_rate': float, 'max_position': int, 'share_size': int}}.
4. The `action` is an int: 0 (Hold), 1 (Buy), 2 (Sell).
5. Ensure the logic (affordability checks, position bounds, commission) exactly matches the Coq code.
"""

    resp = llm.invoke([{"role": "user", "content": translation_prompt}])
    ref_code = resp.content.replace("```python", "").replace("```", "").strip()

    # 3. Differential Testing (Fuzzing)
    print("[verifier] Running differential fuzzer against generated agent...")
    
    # Load the generated agent and the reference step
    test_locals = {}
    try:
        # We need gymnasiym and numpy in the namespace
        import gymnasium as gym
        from gymnasium import spaces
        exec(generated_code, {"gym": gym, "spaces": spaces, "np": np, "torch": None, "optim": None, "nn": None, "pd": None, "random": random, "deque": None, "os": os}, test_locals)
        exec(ref_code, {}, test_locals)
    except Exception as e:
        return {**state, "proof_passed": False, "proof_error": f"Failed to load code: {str(e)}", "retry_feedback": f"Python loading error: {str(e)}"}

    TradingEnv = test_locals.get("TradingEnv")
    reference_step = test_locals.get("reference_step")

    if not TradingEnv or not reference_step:
        return {**state, "proof_passed": False, "proof_error": "Could not find TradingEnv or reference_step"}

    # Fuzz parameters
    params = {
        "commission_rate": 0.001,
        "max_position": 100,
        "share_size": 1
    }
    
    # Mock data for env
    mock_data = np.zeros((100, 5))
    mock_data[:, 3] = 100.0 # Price is 100
    
    env = TradingEnv(data=mock_data, initial_cash=10000.0, **params)
    
    errors = []
    for i in range(100):
        obs, info = env.reset()
        # Random step
        action = random.randint(0, 2)
        
        # State before
        cash_pre = env._cash
        shares_pre = env._shares_held
        price = env._get_current_price()
        
        # Agent execution
        next_obs, reward, term, trunc, info = env.step(action)
        cash_post = env._cash
        shares_post = env._shares_held
        
        # Reference execution
        ref_state = {"cash": float(cash_pre), "shares_held": int(shares_pre)}
        ref_params = params
        new_ref_state = reference_step(ref_params, ref_state, float(price), action)
        
        # Compare
        if abs(cash_post - new_ref_state['cash']) > 1e-6 or shares_post != new_ref_state['shares_held']:
            err = (f"Mismatch at step {i}, Action {action}\n"
                   f"Inputs: Cash={cash_pre}, Shares={shares_pre}, Price={price}\n"
                   f"Agent: Cash={cash_post}, Shares={shares_post}\n"
                   f"Ref:   Cash={new_ref_state['cash']}, Shares={new_ref_state['shares_held']}")
            errors.append(err)
            if len(errors) >= 3: break

    if errors:
        feedback = "Your agent's step logic does NOT match the verified Coq specification!\n\n" + "\n\n".join(errors)
        print(f"[verifier] ✗ Verification failed: {len(errors)} mismatches found")
        return {**state, "proof_passed": False, "proof_error": feedback, "retry_feedback": feedback, "current_phase": "code"}

    print("[verifier] ✓ All 100 differential tests passed!")
    return {**state, "proof_passed": True, "current_phase": "push"}
