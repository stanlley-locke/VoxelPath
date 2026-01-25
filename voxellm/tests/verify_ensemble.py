
import sys
import os
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from voxellm.core.ensemble import EnsembleEngine, ModelRouter

def test_ensemble():
    print("[TEST] Testing Model Router...")
    router = ModelRouter()
    
    cases = [
        ("How to define a class in Python", "coding"),
        ("The patient has fever symptoms", "medical"),
        ("Once upon a time in a castle", "creative"),
        ("What is the capital of France?", "general")
    ]
    
    for prompt, expected in cases:
        result = router.route(prompt)
        print(f"Prompt: '{prompt[:20]}...' -> Route: {result} (Expected: {expected})")
        if result != expected:
            print("FAIL: Routing mismatch")
            return
            
    print("PASS: Router logic verified.")
    
    print("\n[TEST] Testing Ensemble Engine...")
    test_dir = "test_voxel_ensemble"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    ensemble = EnsembleEngine(base_dir=test_dir)
    
    # Test generation routing
    print("[TEST] Generating medical response...")
    # This should trigger loading of 'medical' model
    ensemble.generate_text("The doctor said diagnostics are needed", max_length=10)
    
    if "medical" in ensemble.engines:
        print("PASS: Medical model loaded.")
    else:
        print("FAIL: Medical model NOT loaded.")
        
    print("[TEST] Generating coding response...")
    ensemble.generate_text("def init function", max_length=10)
    
    if "coding" in ensemble.engines:
        print("PASS: Coding model loaded.")
    else:
        print("FAIL: Coding model NOT loaded.")
        
    print("[TEST] Ensemble verification complete.")

if __name__ == "__main__":
    test_ensemble()
