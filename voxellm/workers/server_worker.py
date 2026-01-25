
import sys
import os
import subprocess
import signal
from PyQt6.QtCore import QThread, pyqtSignal

class ServerWorker(QThread):
    """
    Background worker to manage the FastAPI/Uvicorn server process.
    """
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(bool) # True = Running, False = Stopped

    def __init__(self, port=8000):
        super().__init__()
        self.port = port
        self.process = None
        self.is_running = False
        
    def run(self):
        """
        Start the server process.
        Note: We use subprocess.Popen instead of uvicorn.run directly 
        to ensure we can kill it cleanly and it doesn't block the GIL in a way that freezes GUI.
        """
        cmd = [sys.executable, "voxellm/server_main.py"]
        
        # Prepare environment
        env = os.environ.copy()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        
        try:
            self.log_signal.emit(f"[SERVER] Starting on port {self.port}...")
            # Use distinct process group for cleaner kill on Unix, 
            # on Windows creationflags=subprocess.CREATE_NEW_PROCESS_GROUP or similiar might be needed?
            # For simplicity, standard Popen.
            
            # We want to capture output line by line
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=project_root # Run from project root
            )
            
            self.is_running = True
            self.status_signal.emit(True)
            
            # Read stdout loop
            while self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    self.log_signal.emit(line.strip())
            
            # If we get here, process died
            self.is_running = False
            self.status_signal.emit(False)
            self.log_signal.emit(f"[SERVER] Process exited with code {self.process.returncode}")
            
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Failed to start server: {e}")
            self.status_signal.emit(False)

    def stop(self):
        """Stop the server process"""
        if self.process and self.is_running:
            self.log_signal.emit("[SERVER] Stopping...")
            # Windows kill
            if sys.platform == 'win32':
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.process.pid)])
            else:
                self.process.terminate()
            
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                
            self.is_running = False
            self.status_signal.emit(False)
            self.log_signal.emit("[SERVER] Stopped.")
