
import sys
import os
from PyQt6.QtWidgets import QApplication

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from voxellm.ui.main_window import VoxelPathProductionUI
from voxellm.core.engine import VoxelPathProductionEngine

# Mock Engine to avoid heavy loading
class MockEngine:
    def __init__(self):
        self.tokenizer = None
    
    def generate_text(self, seed_text, stop_sequences=None, **kwargs):
        print(f"[MOCK] Generating with seed: '{seed_text}'")
        print(f"[MOCK] Stop sequences: {stop_sequences}")
        return "Generated response\n", [], {}

def test_raw_chat_logic():
    app = QApplication(sys.argv)
    
    # Initialize UI (headless-ish)
    window = VoxelPathProductionUI()
    window.engine = MockEngine() # Swap real engine with mock
    
    print("\n--- Test 1: Standard Mode ---")
    window.cb_raw_mode.setChecked(False)
    window.txt_chat_input.setText("Hello Standard")
    window.conversation_history = [] # Reset
    window.send_chat_message()
    
    print("\n--- Test 2: Raw Mode ---")
    window.cb_raw_mode.setChecked(True)
    window.txt_chat_input.setText("Hello Raw")
    window.conversation_history = [{"role": "User", "content": "Previous User"}, {"role": "AI", "content": "Previous AI"}]
    window.send_chat_message()

if __name__ == "__main__":
    test_raw_chat_logic()
