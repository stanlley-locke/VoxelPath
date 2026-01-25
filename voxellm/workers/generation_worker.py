from PyQt6.QtCore import QThread, pyqtSignal
import time
import traceback

from ..core.engine import VoxelPathProductionEngine

class GenerationWorker(QThread):
    """Background text generation worker with comprehensive metrics"""
    generation_complete = pyqtSignal(str, list, dict)  # (generated_text, trajectory, metrics)
    generation_progress = pyqtSignal(int, str)  # (progress_percent, status)
    
    def __init__(self,
                engine: VoxelPathProductionEngine,
                seed_text: str,
                max_length: int = 200,
                temperature: float = 0.85,
                top_p: float = 0.9,
                repetition_penalty: float = 1.2,
                use_astar: bool = False,
                use_momentum: bool = True,
                min_length: int = 50):
        super().__init__()
        self.engine = engine
        self.seed_text = seed_text
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.use_astar = use_astar
        self.use_momentum = use_momentum
        self.min_length = min_length
        
    def run(self):
        """Execute generation"""
        try:
            self.generation_progress.emit(10, "Initializing generation...")
            
            # Generate text
            start_time = time.time()
            generated_text, trajectory, metrics = self.engine.generate_text(
                seed_text=self.seed_text,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                use_astar=self.use_astar,
                use_momentum=self.use_momentum,
                min_length=self.min_length
            )
            generation_time = time.time() - start_time
            
            # Convert metrics to dictionary
            metrics_dict = {
                'perplexity': metrics.perplexity,
                'coherence_score': metrics.coherence_score,
                'diversity_score': metrics.diversity_score,
                'repetition_rate': metrics.repetition_rate,
                'avg_word_length': metrics.avg_word_length,
                'sentence_complexity': metrics.sentence_complexity,
                'grammatical_correctness': metrics.grammatical_correctness,
                'topic_consistency': metrics.topic_consistency,
                'generation_time': generation_time,
                'character_count': len(generated_text),
                'token_count': len(self.engine.tokenizer.tokenize(generated_text))
            }
            
            self.generation_progress.emit(100, "Generation complete!")
            self.generation_complete.emit(generated_text, trajectory, metrics_dict)
            
        except Exception as e:
            error_msg = f"Generation error: {str(e)}\n{traceback.format_exc()}"
            self.generation_complete.emit(error_msg, [], {})
