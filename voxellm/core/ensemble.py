
import re
from typing import Dict, List, Optional
import os
from .engine import VoxelPathProductionEngine

class ModelRouter:
    """Decides which specialized model should handle a prompt"""
    
    def __init__(self):
        self.rules = {
            "medical": [r"doctor", r"medicine", r"health", r"patient", r"symptom", r"diagnosis", r"treatment"],
            "coding": [r"python", r"java", r"code", r"function", r"class", r"import", r"debugging", r"error"],
            "creative": [r"story", r"tale", r"once upon", r"novel", r"poem", r"character"]
        }
        
    def route(self, prompt: str) -> str:
        """Return the model key for the prompt (default: 'general')"""
        prompt_lower = prompt.lower()
        
        scores = {key: 0 for key in self.rules}
        
        for key, patterns in self.rules.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    scores[key] += 1
                    
        # Find best match
        best_model = max(scores.items(), key=lambda x: x[1])
        if best_model[1] > 0:
            return best_model[0]
            
        return "general"

class EnsembleEngine:
    """Manages multiple specialized VoxelLM engines"""
    
    def __init__(self, base_dir: str = "voxel_ensemble"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        self.router = ModelRouter()
        self.engines: Dict[str, VoxelPathProductionEngine] = {}
        self.active_model = "general"
        
        # Always initialize general model
        self.load_model("general")
        
    def load_model(self, model_name: str):
        """Lazy load a model"""
        if model_name in self.engines:
            return
            
        print(f"[ENSEMBLE] Loading specialized model: {model_name}")
        model_path = os.path.join(self.base_dir, model_name)
        self.engines[model_name] = VoxelPathProductionEngine(model_dir=model_path)
        
    def generate_text(self, seed_text: str, **kwargs):
        """Route generation to the appropriate model"""
        # Route
        target_model = self.router.route(seed_text)
        
        if target_model != self.active_model:
            print(f"[ENSEMBLE] Switching context: {self.active_model} -> {target_model}")
            self.load_model(target_model)
            self.active_model = target_model
            
        engine = self.engines[self.active_model]
        return engine.generate_text(seed_text, **kwargs)
        
    def train_specialized(self, text: str, category: str, batch_size: int = 5000):
        """Train a specific expert model"""
        self.load_model(category)
        print(f"[ENSEMBLE] Training expert: {category}")
        return self.engines[category].train_corpus(text, batch_size)
