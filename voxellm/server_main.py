"""
VoxelLM API Server Entry Point
Run this to start the FastAPI server
"""

import uvicorn
import os
import sys

def main():
    """Run the VoxelLM API Server"""
    print("[SERVER] Starting VoxelLM Neural/Voxel Server...")
    print("[SERVER] Host: http://localhost:8000")
    print("[SERVER] Docs: http://localhost:8000/docs")
    
    # Add project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        # Import app directly to catch import errors early
        from voxellm.server.app import app
        
        # Run with port retry logic
        port = 8000
        max_retries = 5
        for i in range(max_retries):
            try:
                print(f"[SERVER] Attempting to bind to port {port}...")
                uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
                break
            except OSError as e:
                # 10048 is 'Address already in use' on Windows
                if e.errno == 10048 or "10048" in str(e):
                    print(f"[SERVER] Port {port} is busy. Trying next port...")
                    port += 1
                else:
                    raise e
        else:
            print("[ERROR] Could not find an available port in range 8000-8005.")
    except ImportError as e:
        print("\n[ERROR] Failed to start server!")
        print(f"Details: {e}")
        print("Please check that dependencies are installed:")
        print("pip install fastapi uvicorn")
        sys.exit(1)

if __name__ == "__main__":
    main()

