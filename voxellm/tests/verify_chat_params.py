
import sys
import os
from PyQt6.QtWidgets import QApplication

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from voxellm.ui.main_window import VoxelPathProductionUI

# Mock Engine
class MockEngine:
    def __init__(self):
        self.tokenizer = None
    
    def generate_text(self, seed_text, temperature=0.7, repetition_penalty=1.2, top_p=0.9, **kwargs):
        print(f"[MOCK] Generating with Temp={temperature}, RepPenalty={repetition_penalty}, TopP={top_p}")
        return "Generated response\n", [], {}

def test_chat_params():
    app = QApplication(sys.argv)
    
    # Initialize UI
    window = VoxelPathProductionUI()
    window.engine = MockEngine() 
    
    # Set slider values
    print("Setting sliders: Temp=1.5, Rep=2.0 (Val=20), TopP=0.5")
    window.slider_temp.setValue(150) # 1.5
    window.slider_rep_penalty.setValue(20) # 2.0
    window.slider_top_p.setValue(50) # 0.5
    
    # Test send
    window.txt_chat_input.setText("Test message")
    window.send_chat_message()
    print("Send complete")

if __name__ == "__main__":
    test_chat_params()
