
import sys
import os
import shutil
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from voxellm.core.engine import VoxelPathProductionEngine
from voxellm.core.structures import NGramLevel

def test_db_engine():
    print("[TEST] Initializing Engine with SQLite Backend...")
    
    # Use a test model directory
    test_dir = "test_voxel_models"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    engine = VoxelPathProductionEngine(model_dir=test_dir)
    
    # Test Training
    print("[TEST] Training on small corpus...")
    corpus = "The quick brown fox jumps over the lazy dog. " * 10
    engine.train_corpus(corpus, batch_size=50)
    
    # Verify DB persistence
    print("[TEST] Verifying persistence...")
    node = engine.storage.get_ngram(('the', 'quick'))
    if node:
        print(f"PASS: Found 'the quick' -> {dict(node.next_words)}")
    else:
        print("FAIL: 'the quick' not found in DB")
        
    # Verify Generation
    print("[TEST] Generating text...")
    text, _, _ = engine.generate_text("The quick", max_length=20)
    print(f"Generated: {text}")
    
    # Verify Optimization
    print("[TEST] Running optimization...")
    engine._optimize_memory()
    print("PASS: Optimization complete")
    
    engine.storage.close()
    # clean up handled by os.rmtree at start of next run usually, keeping for inspection

if __name__ == "__main__":
    test_db_engine()
