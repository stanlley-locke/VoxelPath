import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QCoreApplication
import qdarktheme

# Add project root to path to ensure imports work if run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from voxellm.ui.main_window import VoxelPathProductionUI
from voxellm.ui.styles import VoxelTheme

def main():
    # Enable High DPI scaling
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("VoxelLM Production System")
    
    # Apply centralized theme
    VoxelTheme.apply_theme(app)
    
    # Global exception handling
    def exception_hook(cls, exception, traceback_obj):
        sys.__excepthook__(cls, exception, traceback_obj)
        
    sys.excepthook = exception_hook
    
    # Create main window
    window = VoxelPathProductionUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
