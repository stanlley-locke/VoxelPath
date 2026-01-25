
import sys
import os
from PyQt6.QtWidgets import QApplication

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from voxellm.ui.styles import VoxelTheme

def test_styles():
    app = QApplication(sys.argv)
    try:
        print("[TEST] Applying light theme...")
        VoxelTheme.apply_theme(app, "light")
        print("[TEST] Light theme OK")
        
        print("[TEST] Applying dark theme...")
        VoxelTheme.apply_theme(app, "dark")
        print("[TEST] Dark theme OK")
        
        print("[SUCCESS] Stylesheet generated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to apply theme: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_styles()
