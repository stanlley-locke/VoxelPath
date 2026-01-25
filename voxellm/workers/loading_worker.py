
import os
import time
from PyQt6.QtCore import QThread, pyqtSignal

class ModelLoaderWorker(QThread):
    """
    Background worker for loading models to prevent UI freezing.
    Also handles syncing to persistent storage.
    """
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(int, str)   # percentage, status message
    
    def __init__(self, engine, model_path):
        super().__init__()
        self.engine = engine
        self.model_path = model_path
        
    def run(self):
        try:
            self.progress.emit(10, "Initializing load...")
            
            # 1. Load the model into memory (this updates engine state)
            # We assume engine.load_model is mostly thread-safe as it replaces dicts
            self.engine.load_model(self.model_path)
            self.progress.emit(60, "Model loaded into memory. Syncing to database...")
            
            # 2. Sync to SQLite for persistence and fallback support
            if hasattr(self.engine, 'sync_to_storage'):
                self.engine.sync_to_storage(
                    progress_callback=lambda p, msg: self.progress.emit(60 + int(p * 0.4), msg)
                )
            else:
                 # Fallback if method missing
                 time.sleep(0.5)

            self.progress.emit(100, "Load complete")
            self.finished.emit(True, f"Successfully loaded model from {os.path.basename(self.model_path)}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e))
