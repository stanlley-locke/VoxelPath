
import sys
import os
from PyQt6.QtWidgets import QApplication

# Add project root (voxelpath) to path, so we can import voxellm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from voxellm.ui.help import AboutDialog, DocumentationDialog

def test_dialogs():
    app = QApplication(sys.argv)
    
    print("[TEST] Instantiating AboutDialog...")
    about = AboutDialog()
    assert about is not None
    print("[TEST] AboutDialog OK")
    
    print("[TEST] Instantiating DocumentationDialog...")
    docs = DocumentationDialog()
    assert docs is not None
    print("[TEST] DocumentationDialog OK")
    
    # Tutorial requires MainWindow, skip for unit test
    
    print("[SUCCESS] Help UI components instantiated successfully.")

if __name__ == "__main__":
    test_dialogs()
