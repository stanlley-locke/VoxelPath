from PyQt6.QtCore import QThread, pyqtSignal
import time
import os
import random
import traceback
import datetime

from ..core.engine import VoxelPathProductionEngine

class TrainingWorker(QThread):
    """Advanced training worker with real-time progress reporting and validation"""
    # Signals
    progress_updated = pyqtSignal(int, str, dict)  # (progress_percent, status_message, metrics)
    training_complete = pyqtSignal(str, dict)  # (completion_message, final_metrics)
    error_occurred = pyqtSignal(str)  # error_message
    validation_metrics = pyqtSignal(dict)  # validation metrics
    
    def __init__(self, engine: VoxelPathProductionEngine, corpus_path: str, validation_path: str = None, **config):
        super().__init__()
        self.engine = engine
        self.corpus_path = corpus_path
        self.validation_path = validation_path
        self.config = config  # Store configuration
        self.is_running = True
        self.total_chars = 0
        self.start_time = time.time()
        self.last_checkpoint = time.time()
        
    def run(self):
        """Main training execution with validation support"""
        try:
            # Read corpus
            self.progress_updated.emit(0, f"Reading corpus: {os.path.basename(self.corpus_path)}", {})
            # Read corpus
            self.progress_updated.emit(0, f"Reading corpus: {os.path.basename(self.corpus_path)}", {})
            import json
            
            if self.corpus_path.lower().endswith(('.jsonl', '.json')):
                corpus_text = ""
                with open(self.corpus_path, 'r', encoding='utf-8', errors='replace') as f:
                    # Check if it's a JSON array (standard JSON) or JSONL
                    first_char = f.read(1)
                    f.seek(0)
                    
                    if first_char == '[':
                        # Standard JSON array
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if "messages" in item:
                                    for msg in item["messages"]:
                                        corpus_text += f"{msg['role']}: {msg['content']}\n"
                                    corpus_text += "\n"
                                elif "prompt" in item and "completion" in item:
                                    corpus_text += f"User: {item['prompt']}\nAI: {item['completion']}\n"
                    else:
                        # JSONL
                        for line in f:
                            try:
                                line = line.strip()
                                if not line: continue
                                item = json.loads(line)
                                if "messages" in item:
                                    for msg in item["messages"]:
                                        corpus_text += f"{msg['role']}: {msg['content']}\n"
                                    corpus_text += "\n"
                                elif "prompt" in item and "completion" in item:
                                    corpus_text += f"User: {item['prompt']}\nAI: {item['completion']}\n"
                            except Exception:
                                continue
            else:
                with open(self.corpus_path, 'r', encoding='utf-8', errors='replace') as f:
                    corpus_text = f.read()
                
            # Read validation corpus if available
            validation_text = None
            if self.validation_path and os.path.exists(self.validation_path):
                with open(self.validation_path, 'r', encoding='utf-8', errors='replace') as f:
                    validation_text = f.read()
                    # Limit validation size
                    validation_text = validation_text[:max(10000, len(validation_text)//10)]
                    
            # Size check with dynamic limits
            file_size = len(corpus_text)
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                self.error_occurred.emit("Corpus too large (max 100MB)")
                return
                
            self.total_chars = file_size
            self.progress_updated.emit(5, f"Processing {file_size:,} characters...", {})
            
            # Training parameters
            batch_size = 50000
            total_batches = (file_size // batch_size) + 1
            
            # Process in chunks
            metrics_history = []
            
            for batch_idx in range(total_batches):
                if not self.is_running:
                    break
                    
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, file_size)
                batch_text = corpus_text[batch_start:batch_end]
                
                # Train on batch
                batch_start_time = time.time()
                metrics = self.engine.train_corpus(
                    batch_text, 
                    batch_size=10000,
                    validation_text=validation_text
                )
                batch_time = time.time() - batch_start_time
                
                # Update progress
                progress = min(95, int(((batch_idx + 1) / total_batches) * 90) + 5)
                elapsed = time.time() - self.start_time
                eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1) if batch_idx > 0 else 0
                
                status = f"Batch {batch_idx+1}/{total_batches}: {len(batch_text):,} chars, "
                status += f"{metrics.training_rate:.0f} t/s, "
                status += f"Loss: {metrics.training_loss:.4f}, "  # Added Loss
                status += f"Ppl: {metrics.perplexity:.2f}, "
                status += f"ETA: {int(eta)}s"
                
                # Convert metrics to dict for signal
                metrics_dict = {
                    'tokens_processed': self.engine.tokens_processed,
                    'ngrams_learned': self.engine.ngrams_learned.copy(),
                    'memory_usage_mb': self.engine.memory_usage_mb,
                    'perplexity': metrics.perplexity,
                    'coherence_score': metrics.coherence_score,
                    'diversity_score': metrics.diversity_score,
                    'training_rate': metrics.training_rate,
                    'batch_time': batch_time,
                    'batch_idx': batch_idx,
                    'total_batches': total_batches
                }
                
                self.progress_updated.emit(progress, status, metrics_dict)
                metrics_history.append(metrics_dict)
                
                # Yield to event loop to keep UI responsive
                self.msleep(10)  # Sleep 10ms to allow GUI updates
                
                # Auto-checkpoint every 5 minutes
                if (time.time() - self.last_checkpoint) > 300:  # 5 minutes
                    self.progress_updated.emit(progress, "Auto-checkpointing...", metrics_dict)
                    self.engine.save_model("auto_checkpoint")
                    self.last_checkpoint = time.time()
                
                # Periodic validation
                if validation_text and (batch_idx + 1) % 3 == 0:
                    self._run_validation(validation_text)
                    
            if self.is_running:
                # Final optimization
                self.progress_updated.emit(96, "Optimizing memory...", {})
                self.engine._optimize_memory()
                
                # Final validation
                if validation_text:
                    self._run_validation(validation_text)
                    
                # Save model
                self.progress_updated.emit(98, "Saving model snapshot...", {})
                save_path = self.engine.save_model()
                
                # Final metrics
                final_metrics = {
                    'tokens_processed': self.engine.tokens_processed,
                    'total_ngrams': sum(self.engine.ngrams_learned.values()),
                    'ngrams_learned': self.engine.ngrams_learned.copy(),
                    'memory_usage_mb': self.engine.memory_usage_mb,
                    'training_time': time.time() - self.start_time,
                    'avg_perplexity': sum(m.get('perplexity', float('inf')) for m in metrics_history) / len(metrics_history) if metrics_history else float('inf')
                }
                
                self.progress_updated.emit(100, "Training complete!", final_metrics)
                self.training_complete.emit(f"Model saved to {save_path}", final_metrics)
                
        except Exception as e:
            self.error_occurred.emit(f"Training error: {str(e)}\n{traceback.format_exc()}")
            
    def _run_validation(self, validation_text: str):
        """Run validation on a sample of text"""
        try:
            # Sample a portion of the validation text
            sample_size = min(5000, len(validation_text))
            start_idx = random.randint(0, max(0, len(validation_text) - sample_size))
            sample_text = validation_text[start_idx:start_idx + sample_size]
            
            # Calculate perplexity
            validation_tokens = self.engine.tokenizer.tokenize(sample_text)
            perplexity = self.engine._calculate_validation_perplexity(validation_tokens)
            
            # Generate sample text
            sample_seed = ' '.join(validation_tokens[:5]) if validation_tokens else "it was the best of times"
            generated_text, _, metrics = self.engine.generate_text(
                seed_text=sample_seed,
                max_length=100,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
            validation_metrics = {
                'perplexity': perplexity,
                'sample_generation': generated_text,
                'coherence_score': metrics.coherence_score,
                'diversity_score': metrics.diversity_score,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.validation_metrics.emit(validation_metrics)
            
        except Exception as e:
            print(f"Validation error: {e}")
            
    def stop(self):
        """Graceful stop"""
        self.is_running = False
