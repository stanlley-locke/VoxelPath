
from PyQt6.QtCore import QThread, pyqtSignal

class RagIngestionWorker(QThread):
    """
    Background worker for RAG document ingestion.
    Prevents UI freeze during embedding generation.
    """
    finished = pyqtSignal(bool, str) # success, message
    
    def __init__(self, engine, text, source_name):
        super().__init__()
        self.engine = engine
        self.text = text
        self.source_name = source_name
        
    def run(self):
        try:
            # This is the heavy blocking call
            self.engine.learn_document(self.text, source_name=self.source_name)
            self.finished.emit(True, f"Successfully ingested: {self.source_name}")
        except Exception as e:
            self.finished.emit(False, str(e))
