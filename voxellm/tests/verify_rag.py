
import sys
import os
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from voxellm.core.engine import VoxelPathProductionEngine
from voxellm.core.rag import VectorStore

def test_rag():
    print("[TEST] Initializing Engine with RAG...")
    
    test_dir = "test_voxel_rag"
    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
        
    engine = VoxelPathProductionEngine(model_dir=test_dir)
    
    # Ingest mock document
    doc = """
    VoxelLM is an advanced AI system that uses 3D geometric positioning to understand language.
    It was created by Antigravity in 2024.
    The system maps words to voxels in a high-dimensional tensor.
    """
    print("[TEST] Ingesting document...")
    engine.learn_document(doc, source_name="manual_test")
    
    # Check if data persisted
    # Since we might not have chromadb installed, we expect graceful fallback or success if installed.
    if engine.rag.has_chroma:
        count = engine.rag.count()
        print(f"[TEST] ChromaDB Count: {count}")
        if count > 0:
            print("PASS: Documents indexed.")
        else:
            print("FAIL: No documents found.")
            
        # Retrieval Test
        print("[TEST] Querying facts...")
        facts = engine.rag.query("Who created VoxelLM?")
        print(f"Results: {facts}")
        if any("Antigravity" in f for f in facts):
            print("PASS: Retrieval successful.")
        else:
            print("FAIL: Retrieval mismatch.")
            
        # Generation Test (with context)
        print("[TEST] Generating text with context...")
        text, _, _ = engine.generate_text("Who created VoxelLM?", max_length=50)
        print(f"Generated: {text}")
        
    else:
        print("[WARN] ChromaDB not installed, skipping RAG logic.")

if __name__ == "__main__":
    test_rag()
