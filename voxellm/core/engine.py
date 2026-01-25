import os
import time
import datetime
import math
import random
import json
import pickle
import gzip
import heapq
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict, deque, Counter

from .structures import (
    NGramLevel, NGramNode, TrajectoryTunnel, GeometricMomentum, 
    TrainingMetrics, TextAnalysisResult
)
from .tokenizer import IntelligentTokenizer
from .storage import SQLiteStorage
from .rag import VectorStore, DocumentProcessor

class VoxelPathProductionEngine:
    """Production-grade stochastic graph-based architecture with training metrics"""
    def __init__(self, voxel_resolution: int = 128, model_dir: str = "voxel_models"):
        # Configuration
        self.resolution = voxel_resolution
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Core data structures
        self.tokenizer = IntelligentTokenizer()
        self.reset_engine()
        
        # Training statistics and history
        self.training_history = []
        self.validation_history = []
        self.start_time = time.time()
        self.last_token_count = 0
        self.last_time = time.time()
        
        # Training configuration
        self.validation_split = 0.1  # 10% of data for validation
        self.early_stopping_patience = 5
        self.best_validation_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def reset_engine(self):
        """Complete engine reset - fresh start"""
        print("[ENGINE] Connecting to SQLite Memory")
        # Persistent Storage
        db_path = os.path.join(self.model_dir, "voxel_brain.db")
        self.storage = SQLiteStorage(db_path)
        
        # Hierarchical N-Gram memory (Hybrid Mode: In-Memory + SQLite)
        self.ngram_memory = {
            NGramLevel.BIGRAM: defaultdict(lambda: NGramNode(context=())),
            NGramLevel.TRIGRAM: defaultdict(lambda: NGramNode(context=())),
            NGramLevel.QUADGRAM: defaultdict(lambda: NGramNode(context=())),
            NGramLevel.QUINTGRAM: defaultdict(lambda: NGramNode(context=())),
        }
        
        # 3D Voxel grid for geometric representation
        self.voxel_tensor = np.zeros(
            (self.resolution, self.resolution, self.resolution),
            dtype=np.float32
        )
        
        # Spatial graph for A* pathfinding
        self.spatial_graph = defaultdict(set)  # position -> {neighbor_positions}
        self.position_mapping = {}  # token -> 3D position
        
        # Advanced features
        self.trajectory_tunnels = {}  # pattern_hash -> TrajectoryTunnel
        self.geometric_momentum = GeometricMomentum(position=(self.resolution//2, self.resolution//2, self.resolution//2))
        self.recent_context = deque(maxlen=50)  # For repetition penalty and coherence
        
        # RAG Knowledge Base
        self.rag = VectorStore(persistence_dir=os.path.join(self.model_dir, "rag_db")) if os.path.isdir(self.model_dir) else VectorStore()
        
        # Training state
        self.is_training = False
        self.current_corpus = None
        self.optimization_scheduled = False
        self.tokens_processed = 0
        self.ngrams_learned = {n: 0 for n in range(2, 6)}
        self.training_time = 0.0
        self.memory_usage_mb = 0.0
        
        print("[ENGINE] Engine reset complete")
        
    def _calculate_3d_position(self, token: str) -> Tuple[int, int, int]:
        """Map token to 3D coordinate using intelligent hashing"""
        if not token:
            return (self.resolution//2, self.resolution//2, self.resolution//2)
            
        # Use first three characters or pad
        padded = (token + '   ')[:3]
        
        # Convert to ASCII values with bounds checking
        x = min(ord(padded[0]), self.resolution - 1) if padded[0] else self.resolution//2
        y = min(ord(padded[1]), self.resolution - 1) if padded[1] else self.resolution//2
        z = min(ord(padded[2]), self.resolution - 1) if padded[2] else self.resolution//2
        
        return (int(x), int(y), int(z))
        
    def train_corpus(self, text: str, batch_size: int = 10000, validation_text: str = None):
        """Advanced training with hierarchical N-Gram learning and validation metrics"""
        start_time = time.time()
        
        # Tokenize corpus
        tokens = self.tokenizer.tokenize(text)
        
        # Show detailed breakdown
        chars_per_token = len(text) / len(tokens) if tokens else 0
        print(f"[TRAIN] ═══════════════════════════════════════════")
        print(f"  Input:      {len(text):,} characters")
        print(f"  Tokenized:  {len(tokens):,} tokens ({chars_per_token:.2f} chars/tok)")
        print(f"[TRAIN] ═══════════════════════════════════════════")
        
        self.tokens_processed += len(tokens)
        
        # Split for validation if needed
        validation_tokens = []
        if validation_text:
            validation_tokens = self.tokenizer.tokenize(validation_text)
            
        # Initialize metrics
        current_metrics = TrainingMetrics()
        current_metrics.tokens_processed = self.tokens_processed
        current_metrics.timestamp = datetime.datetime.now()
        
        # Process in batches
        total_batches = max(1, (len(tokens) + batch_size - 1) // batch_size)  # Ceiling division
        tokens_per_second = 0
        cumulative_ngrams = {n: 0 for n in range(2, 6)}
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(tokens))
            batch_tokens = tokens[batch_start:batch_end]
            
            if not batch_tokens:
                continue
                
            # Update timing for rate calculation
            current_time = time.time()
            time_elapsed = current_time - self.last_time
            tokens_in_batch = len(batch_tokens)
            
            if time_elapsed > 0:
                tokens_per_second = tokens_in_batch / time_elapsed
                current_metrics.training_rate = tokens_per_second
                
            self.last_time = current_time
            self.last_token_count = tokens_in_batch
            
            # Process batch
            batch_start_time = time.time()
            training_loss, batch_ngrams = self._process_batch(batch_tokens)
            batch_time = time.time() - batch_start_time
            
            # Track cumulative N-grams
            for level, count in batch_ngrams.items():
                cumulative_ngrams[level] += count
            
            # Update metrics
            current_metrics.epoch = batch_idx + 1
            current_metrics.tokens_processed = self.tokens_processed
            current_metrics.training_loss = training_loss
            current_metrics.ngrams_learned = cumulative_ngrams.copy()
            
            # Calculate perplexity on validation set if available
            if validation_tokens and len(validation_tokens) > 100:
                current_metrics.perplexity = self._calculate_validation_perplexity(validation_tokens[:200])
                current_metrics.validation_loss = current_metrics.perplexity
                
            # Memory optimization every 5 batches
            if batch_idx % 5 == 0 and batch_idx > 0:
                self._optimize_memory()
                current_metrics.memory_usage_mb = self.memory_usage_mb
                
            # Text quality metrics
            if batch_tokens:
                analysis = self.tokenizer.analyze_text(batch_tokens[:min(100, len(batch_tokens))])
                current_metrics.coherence_score = analysis.coherence_score
                current_metrics.diversity_score = analysis.diversity_score
                
            # Save metrics history
            self.training_history.append(current_metrics.to_dict())
            
            # Print progress with enhanced metrics
            progress = (batch_idx + 1) / total_batches * 100
            total_ngrams_batch = sum(batch_ngrams.values())
            
            # Per-level breakdown
            ngram_breakdown = ", ".join([f"{n}g:{batch_ngrams[n]:,}" for n in sorted(batch_ngrams.keys())])
            
            print(f"[TRAIN] Batch {batch_idx+1}/{total_batches} ({progress:.1f}%):")
            print(f"  Tokens:  {len(batch_tokens):,} @ {tokens_per_second:.0f} tok/s")
            print(f"  Loss:    {training_loss:.4f}, PPL: {current_metrics.perplexity:.2f}")
            print(f"  N-grams: +{total_ngrams_batch:,} ({ngram_breakdown})")
            
            # Create new metrics object for next batch
            current_metrics = TrainingMetrics()
            current_metrics.tokens_processed = self.tokens_processed
            current_metrics.timestamp = datetime.datetime.now()
            
        # Final optimization
        self._optimize_memory()
        
        # Final metrics with real counts
        final_metrics = TrainingMetrics()
        final_metrics.epoch = total_batches
        final_metrics.tokens_processed = self.tokens_processed
        final_metrics.ngrams_learned = self._count_ngrams_in_memory()
        final_metrics.memory_usage_mb = self.memory_usage_mb
        final_metrics.training_rate = tokens_per_second
        
        if validation_tokens:
            final_metrics.perplexity = self._calculate_validation_perplexity(validation_tokens)
            final_metrics.validation_loss = final_metrics.perplexity
            
        self.training_history.append(final_metrics.to_dict())
        self.training_time += time.time() - start_time
        
        # Show detailed completion summary
        total_ngrams = sum(final_metrics.ngrams_learned.values())
        ngram_summary = ", ".join([f"{n}g:{final_metrics.ngrams_learned.get(n, 0):,}" 
                                   for n in range(2, 6)])
        duration = time.time() - start_time
        avg_speed = len(tokens) / duration if duration > 0 else 0
        
        print(f"[TRAIN] ═══════════════════════════════════════════")
        print(f"[TRAIN] Training Complete!")
        print(f"[TRAIN] ───────────────────────────────────────────")
        print(f"  Duration:    {duration:.1f}s ({avg_speed:.0f} tok/s avg)")
        print(f"  Total N-grams: {total_ngrams:,}")
        print(f"    Breakdown:   {ngram_summary}")
        print(f"  Final Loss:    {final_metrics.training_loss:.4f}")
        print(f"  Final PPL:     {final_metrics.perplexity:.2f}")
        print(f"  Memory:        {final_metrics.memory_usage_mb:.1f} MB")
        print(f"[TRAIN] ═══════════════════════════════════════════")
              
        return final_metrics
        
    def _process_batch(self, tokens: List[str]) -> Tuple[float, Dict[int, int]]:
        """Process a single batch of tokens for training, returns (avg_loss, ngrams_per_level)"""
        loss_count = 0
        total_loss = 0.0
        ngrams_added = {n: 0 for n in range(2, 6)}
        
        # Defer DB commits - batch them at end
        pending_transitions = []
        
        # Train N-Grams from 2 to 5
        for n in range(2, 6):
            ngram_level = NGramLevel(n)
            for i in range(len(tokens) - n + 1):
                context = tuple(tokens[i:i+n-1])
                next_word = tokens[i+n-1]
                
                # Calculate online training loss (sample 10% for performance)
                if i % 10 == 0:
                    probs = self.recursive_backoff(context)
                    if probs and next_word in probs:
                        prob = probs[next_word]
                        if prob > 0:
                            loss = -math.log(prob)
                            total_loss += loss
                            loss_count += 1
                
                # Update in-memory N-grams (FAST)
                if ngram_level not in self.ngram_memory:
                    self.ngram_memory[ngram_level] = defaultdict(lambda: NGramNode(context=()))
                
                if context not in self.ngram_memory[ngram_level]:
                    self.ngram_memory[ngram_level][context] = NGramNode(context=context)
                    ngrams_added[n] += 1
                
                self.ngram_memory[ngram_level][context].add_transition(next_word, 1.0)
                
                
                # Batch DB writes
                pending_transitions.append((context, next_word, 1.0))
        
        # OPTIMIZATION: Batch commit all transitions at once (eliminates per-transition overhead)
        if pending_transitions and hasattr(self.storage, 'batch_add_transitions'):
            self.storage.batch_add_transitions(pending_transitions)
        
        # Update training metrics
        if loss_count > 0:
            avg_loss = total_loss / loss_count
        else:
            avg_loss = 0.0
            
        return avg_loss, ngrams_added
    
    def _count_ngrams_in_memory(self) -> Dict[int, int]:
        """Count N-grams from hybrid memory (RAM + DB if needed)"""
        counts = {}
        
        # First check in-memory (fast)
        for level, nodes in self.ngram_memory.items():
            counts[level.value] = len(nodes)
        
        # Update self.ngrams_learned for persistence
        for n, count in counts.items():
            self.ngrams_learned[n] = count
        
        return counts
            
    def _calculate_validation_perplexity(self, validation_tokens: List[str]) -> float:
        """Calculate perplexity on validation tokens"""
        if len(validation_tokens) < 10:
            return float('inf')
            
        total_log_prob = 0.0
        total_predictions = 0
        
        # Use a sliding window approach
        for i in range(min(5, len(validation_tokens) - 5)):
            # Get context and next word
            context = tuple(validation_tokens[i:i+4])  # Up to 5-gram context
            next_word = validation_tokens[i+4]
            
            # Get probabilities
            probabilities = self.recursive_backoff(context)
            
            if probabilities and next_word in probabilities:
                prob = probabilities[next_word]
                if prob > 0:
                    total_log_prob += math.log(prob)
                    total_predictions += 1
                    
        if total_predictions == 0:
            return float('inf')
            
        avg_log_prob = total_log_prob / total_predictions
        perplexity = math.exp(-avg_log_prob)
        return perplexity
        
    def recursive_backoff(self, context: Tuple[str, ...]) -> Dict[str, float]:
        """
        Intelligent backoff from 5-gram down to 2-gram
        Returns weighted combination of probabilities from all levels
        """
        if not context:
            return {}
            
        probabilities = defaultdict(float)
        total_weight = 0.0
        
        # Try from 5-gram down to 2-gram
        for n in range(5, 1, -1):
            if len(context) >= n - 1:
                ngram_level = NGramLevel(n)
                context_slice = context[-(n-1):] if n > 1 else ()
                
                # [OPTIMIZATION] Check in-memory fast path first
                node = None
                if ngram_level in self.ngram_memory:
                    if context_slice in self.ngram_memory[ngram_level]:
                        node = self.ngram_memory[ngram_level][context_slice]
                
                # Fallback to storage if not in memory (hybrid mode)
                if not node:
                    node = self.storage.get_ngram(context_slice)

                if node and node.next_words:
                    # Weight decreases with N-Gram level (higher N gets more weight)
                    weight = 2 ** (n - 2)  # 5-gram: 8, 4-gram: 4, 3-gram: 2, 2-gram: 1
                    for word, prob in node.get_logits().items():
                        probabilities[word] += prob * weight
                    total_weight += weight
                    
        # Normalize probabilities
        if total_weight > 0:
            return {w: p / total_weight for w, p in probabilities.items()}
        return {}
        
    def apply_repetition_penalty(self,
                                probabilities: Dict[str, float],
                                recent_tokens: List[str],
                                penalty_strength: float = 1.2) -> Dict[str, float]:
        """
        Dynamic repetition penalty based on recent context
        """
        if not recent_tokens or not probabilities:
            return probabilities
            
        penalized_probs = probabilities.copy()
        
        # Frequency-based penalty
        token_freq = Counter(recent_tokens)
        max_freq = max(token_freq.values()) if token_freq else 1
        
        for word in list(penalized_probs.keys()):
            freq = token_freq.get(word, 0)
            if freq > 0:
                # Dynamic penalty based on frequency and recency
                recency_factor = 1.0 + (freq / max_freq) * 0.5
                penalty = (penalty_strength ** freq) * recency_factor
                penalized_probs[word] /= penalty
                
        # Apply context coherence penalty
        if len(recent_tokens) > 3:
            last_words = recent_tokens[-3:]
            for word in list(penalized_probs.keys()):
                # Penalize words that don't match the semantic field of recent context
                if word[0] != last_words[-1][0] and word[-1] != last_words[-1][-1]:
                    # Very basic semantic field matching
                    if word not in ' '.join(last_words):
                        penalized_probs[word] *= 0.9
                        
        # Renormalize
        total = sum(penalized_probs.values())
        if total > 0:
            return {w: p / total for w, p in penalized_probs.items()}
        return probabilities
        
    def a_star_search(self,
                     start_pos: Tuple[int, int, int],
                     target_chars: Optional[List[str]] = None,
                     max_iterations: int = 1000) -> List[Tuple[int, int, int]]:
        """
        A* pathfinding through voxel space with boundary checking
        """
        if target_chars is None:
            target_chars = [' ', '.', '!', '?', '\n', ',', ';', ':']
            
        # Ensure all target positions are within bounds
        target_positions = []
        for c in target_chars:
            pos = self._calculate_3d_position(c)
            if (0 <= pos[0] < self.resolution and 
                0 <= pos[1] < self.resolution and 
                0 <= pos[2] < self.resolution):
                target_positions.append(pos)
                
        if not target_positions:
            return [start_pos]
            
        # Ensure start position is within bounds
        start_pos = (
            min(max(0, start_pos[0]), self.resolution - 1),
            min(max(0, start_pos[1]), self.resolution - 1),
            min(max(0, start_pos[2]), self.resolution - 1)
        )
        
        # Priority queue: (f_score, g_score, position, path)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_pos, [start_pos]))
        g_scores = {start_pos: 0}
        visited = set([start_pos])
        iterations = 0
        
        while open_set and iterations < max_iterations:
            f_score, g_score, current, path = heapq.heappop(open_set)
            iterations += 1
            
            # Check if reached target
            if current in target_positions:
                return path
                
            # Explore neighbors from spatial graph
            neighbors = self.spatial_graph.get(current, set())
            
            # Add nearby voxels based on voxel density
            if not neighbors and iterations < 100:  # Only do this early in search
                radius = 5
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        for dz in range(-radius, radius + 1):
                            if dx == dy == dz == 0:
                                continue
                                
                            nx, ny, nz = current[0] + dx, current[1] + dy, current[2] + dz
                            
                            # Check bounds
                            if not (0 <= nx < self.resolution and 
                                    0 <= ny < self.resolution and 
                                    0 <= nz < self.resolution):
                                continue
                                
                            # Only consider positions with non-zero density
                            if self.voxel_tensor[nx, ny, nz] > 0:
                                neighbors.add((nx, ny, nz))
                                
            for neighbor in neighbors:
                # Skip if out of bounds
                if not (0 <= neighbor[0] < self.resolution and 
                        0 <= neighbor[1] < self.resolution and 
                        0 <= neighbor[2] < self.resolution):
                    continue
                    
                if neighbor in visited:
                    continue
                    
                # Cost based on voxel density (higher density = lower cost)
                density = self.voxel_tensor[neighbor] + 1e-6  # Add epsilon to avoid division by zero
                tentative_g = g_score + (1.0 / density)
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    
                    # Heuristic: Euclidean distance to nearest target
                    distances = []
                    for t in target_positions:
                        try:
                            dist = (math.sqrt((neighbor[0]-t[0])**2 + 
                                           (neighbor[1]-t[1])**2 + 
                                           (neighbor[2]-t[2])**2))
                            distances.append(dist)
                        except Exception:
                            continue
                            
                    h_score = min(distances) if distances else 0
                    f_score_new = tentative_g + h_score
                    
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f_score_new, tentative_g, neighbor, new_path))
                    visited.add(neighbor)
                    
        # Return best path found
        return path
        
    def learn_document(self, text: str, source_name: str = "user_upload"):
        """Ingest a document into the RAG knowledge base"""
        chunks = DocumentProcessor.chunk_text(text)
        if chunks:
            metadatas = [{"source": source_name, "chunk_id": i} for i in range(len(chunks))]
            self.rag.add_documents(chunks, metadatas=metadatas)
            print(f"[ENGINE] Learned {len(chunks)} facts from {source_name}")

    def generate_text(self,
                     seed_text: str,
                     max_length: int = 200,
                     temperature: float = 0.85,
                     top_p: float = 0.9,
                     repetition_penalty: float = 1.2,
                     use_astar: bool = False,
                     use_momentum: bool = True,
                     min_length: int = 50,
                     stop_sequences: List[str] = None) -> Tuple[str, List[Tuple[int, int, int]], TextAnalysisResult]:
        """
        Advanced text generation with all features enabled and comprehensive metrics
        """
        # RAG Context Injection
        rag_context = ""
        try:
            # Query knowledge base for relevant facts
            facts = self.rag.query(seed_text, n_results=2)
            if facts:
                print(f"[RAG] Found context: {facts[0][:50]}...")
                rag_context = "Context: " + " ".join(facts) + "\n\n"
        except Exception as e: 
            # RAG failure should not stop generation
            print(f"[RAG] Query failed (or disabled): {e}")

        # Combine Context + Prompt
        # We don't want to output the context, just condition on it.
        # So we prime the tokenizer with it, but generated_tokens start from seed.
        
        full_prompt = rag_context + seed_text
        seed_tokens = self.tokenizer.tokenize(full_prompt)
        
        # We want to output only what follows seed_text, but we need the context in history
        # If we use full_prompt tokens as seed, the engine will assume they are "generated" if we aren't careful?
        # No, seed tokens are just the starting context.
        # But if we return user output, we usually don't want to repeat the injected context?
        # Actually, generated_tokens = seed_tokens.copy() is standard here.
        
        if len(seed_tokens) < 1:
            seed_tokens = ['it', 'was', 'the', 'best', 'of', 'times']
            
        generated_tokens = seed_tokens.copy()
        trajectory = []
        analysis_metrics = TextAnalysisResult()
        
        # Track recent context for repetition penalty
        recent_context = deque(maxlen=30)
        recent_context.extend(seed_tokens)
        
        # Get starting position
        start_pos = self._calculate_3d_position(seed_tokens[-1]) if seed_tokens else (0, 0, 0)
        
        for i in range(max_length):
            # Get context (up to 4 previous tokens for 5-gram)
            context = tuple(generated_tokens[-4:])
            
            # Check for trajectory tunnels
            tunnel_key = hash(' '.join(context))
            if tunnel_key in self.trajectory_tunnels:
                tunnel = self.trajectory_tunnels[tunnel_key]
                if tunnel.should_activate():
                    # Use tunnel continuation
                    continuation_tokens = self.tokenizer.tokenize(tunnel.continuation)
                    num_tokens = min(5, len(continuation_tokens))
                    generated_tokens.extend(continuation_tokens[:num_tokens])
                    recent_context.extend(continuation_tokens[:num_tokens])
                    
                    # Add positions to trajectory
                    for token in continuation_tokens[:num_tokens]:
                        pos = self._calculate_3d_position(token)
                        trajectory.append(pos)
                        
                    # Update momentum
                    if use_momentum and trajectory:
                        self.geometric_momentum.update(trajectory[-1])
                    continue
                    
            # Get probabilities with recursive backoff
            probabilities = self.recursive_backoff(context)
            
            # If no probabilities found, use fallback strategy
            # If no probabilities found, use fallback strategy
            if not probabilities:
                # Try simpler context
                if len(context) > 1:
                    probabilities = self.recursive_backoff(context[-1:])
                    
                # Still nothing? Use most common words
                if not probabilities:
                    # 1. Try DB first
                    probabilities = self.storage.get_most_common_words(10)
                    
                    # 2. If DB empty/failed, try in-memory stats
                    if not probabilities:
                        probabilities = self._get_fallback_probabilities()
                    
                    # 3. Ultimate fallback
                    if not probabilities:
                        probabilities = {'the': 0.3, 'and': 0.2, 'of': 0.15, 'to': 0.1, 'a': 0.08, 'in': 0.07, 'is': 0.05, 'it': 0.05}
                        
            # Apply repetition penalty
            probabilities = self.apply_repetition_penalty(
                probabilities,
                list(recent_context)[-15:],  # Last 15 tokens
                repetition_penalty
            )
            
            # Apply temperature
            if temperature != 1.0 and probabilities:
                temp_probs = {}
                temp_sum = 0.0
                
                for word, prob in probabilities.items():
                    # Handle zero probabilities
                    if prob <= 0:
                        continue
                        
                    # Apply temperature scaling
                    scaled = math.exp(math.log(prob) / temperature)
                    temp_probs[word] = scaled
                    temp_sum += scaled
                    
                # Normalize
                if temp_sum > 0:
                    probabilities = {w: p/temp_sum for w, p in temp_probs.items()}
                    
            # Apply top-p (nucleus) sampling
            if top_p < 1.0 and probabilities:
                # Sort probabilities
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                
                # Calculate cumulative probabilities
                cumulative = 0.0
                cutoff_index = 0
                
                for i, (word, prob) in enumerate(sorted_probs):
                    cumulative += prob
                    if cumulative >= top_p:
                        cutoff_index = i + 1
                        break
                        
                # Keep only top-p words
                kept_words = sorted_probs[:cutoff_index]
                
                # Renormalize
                total = sum(prob for _, prob in kept_words)
                if total > 0:
                    probabilities = {word: prob/total for word, prob in kept_words}
                    
            # Sample from distribution
            if probabilities:
                words = list(probabilities.keys())
                probs = list(probabilities.values())
                
                try:
                    # Use numpy for better random sampling
                    next_word = np.random.choice(words, p=probs)
                except ValueError:
                    # Fallback if probabilities don't sum to 1
                    next_word = random.choices(words, weights=probs, k=1)[0]
            else:
                # Fallback to common words
                next_word = random.choice(['the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'that', 'was'])
                
            # Update context
            generated_tokens.append(next_word)
            recent_context.append(next_word)
            
            # Update trajectory
            pos = self._calculate_3d_position(next_word)
            trajectory.append(pos)
            
            # Update geometric momentum
            if use_momentum and trajectory:
                self.geometric_momentum.update(pos)

            # Check for stop sequences (only in new content)
            if stop_sequences and len(generated_tokens) > len(seed_tokens):
                # Get only the new tokens + a bit of context for boundary overlap
                # We need enough context to cover the longest stop sequence
                new_token_count = len(generated_tokens) - len(seed_tokens)
                
                # Check effectively the "new" text, but allow for boundary crossing
                # by taking new tokens plus 5 previous ones (if available)
                lookback = min(len(generated_tokens), new_token_count + 5)
                check_tokens = generated_tokens[-lookback:]
                check_text = self.tokenizer.detokenize(check_tokens)
                
                # But we must ensure we don't match the prompt part again
                # A safer way: detokenize strictly the new tokens?
                # But "User:" might be split across prompt end and new token token? Unlikely with this tokenizer.
                # Actually, simply checking if it appears in the *newly generated* part is safest.
                
                new_text = self.tokenizer.detokenize(generated_tokens[len(seed_tokens):])
                
                stop_found = False
                for seq in stop_sequences:
                    if seq in new_text:
                        stop_found = True
                        break
                if stop_found:
                    break
                    
            # Check for natural ending points
            if i >= min_length - 1 and next_word in ['.', '!', '?', '\n'] and not stop_sequences:
                break
                
        # Convert tokens back to text
                
        # Convert tokens back to text
        generated_text = self.tokenizer.detokenize(generated_tokens)
        
        # Strip RAG context from output if present (simple heuristic)
        if rag_context and generated_text.startswith(rag_context.strip()):
            # This is tricky because detokenization might change spacing. 
            # Generally users want to see the whole conversation including context?
            # Or better, let's just return what was generated *after* the prompt?
            # The function signature returns full text usually.
            pass
        
        # Analyze the generated text
        analysis_metrics = self.tokenizer.analyze_text(generated_tokens)
        
        return generated_text, trajectory, analysis_metrics
        
    def _optimize_memory(self):
        """Memory optimization and cleanup with metrics tracking"""
        # With SQLite backend, we just need to manage the cache and commit
        
        start_time = time.time()
        
        # Commit transactions
        self.storage.commit()
        
        # Clear/Prune LRU cache to free RAM
        # In this simple implementation we just clear it if it gets too big
        # But SQLiteStorage has internal LRU logic, so we can trust it?
        # Actually, let's explicitly trigger a garbage collection of the Python objects
        # if the cache is huge.
        
        # Update memory usage estimate
        self._update_memory_stats()
        self.last_optimization = datetime.datetime.now()
        
        optimization_time = time.time() - start_time
        print(f"[OPTIMIZE] Database committed in {optimization_time:.4f}s. "
              f"Current usage: {self.memory_usage_mb:.1f} MB")
              
    def _update_memory_stats(self):
        """Calculate current memory usage accurately"""
        # Estimate N-Gram memory (Just the cache size)
        ngram_memory = len(self.storage._cache) * 500 # Approx bytes per node
        
        # Voxel tensor memory (float32)
        voxel_memory = self.voxel_tensor.nbytes
                    
        # Voxel tensor memory (float32)
        voxel_memory = self.voxel_tensor.nbytes
        
        # Graph memory
        graph_memory = 0
        for pos, neighbors in self.spatial_graph.items():
            graph_memory += 24  # Position tuple
            graph_memory += len(neighbors) * 24  # Each neighbor reference
            
        # Trajectory tunnels
        tunnel_memory = 0
        for tunnel in self.trajectory_tunnels.values():
            tunnel_memory += len(tunnel.pattern) + len(tunnel.continuation) + 100  # Overhead
            
        total_bytes = ngram_memory + voxel_memory + graph_memory + tunnel_memory
        self.memory_usage_mb = total_bytes / (1024 * 1024)
        
    def identify_concepts(self, top_n: int = 10) -> List[Dict]:
        """
        Identify 'concepts' based on high-density voxel clusters
        Returns list of {position, density, top_words}
        """
        concepts = []
        
        # Find local maxima in voxel tensor
        # Simplified approach: finding top N distinct peaks
        peaks = []
        
        # Get indices of voxels with density > mean + 2*std
        mean_density = np.mean(self.voxel_tensor)
        std_density = np.std(self.voxel_tensor)
        threshold = mean_density + 2 * std_density
        
        indices = np.argwhere(self.voxel_tensor > threshold)
        
        for idx in indices:
             x, y, z = idx
             density = self.voxel_tensor[x, y, z]
             peaks.append(((x, y, z), density))
             
        # Sort by density
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Select distinct peaks (simple suppression)
        selected_peaks = []
        for peak_pos, density in peaks:
            is_distinct = True
            for sel_pos, _ in selected_peaks:
                dist = math.sqrt(sum((a-b)**2 for a, b in zip(peak_pos, sel_pos)))
                if dist < 10:  # Minimum distance between concepts
                    is_distinct = False
                    break
            
            if is_distinct:
                selected_peaks.append((peak_pos, density))
                if len(selected_peaks) >= top_n:
                    break
                    
        # Identify words associated with these peaks
        # This is a reverse lookup, which is slow, so we approximate
        # by checking a sample of high-frequency N-grams
        
        for pos, density in selected_peaks:
            concept_words = []
            
            # Search nearby nodes
            for level in self.ngram_memory.values():
                for node in level.values():
                    if not node.vector_position:
                        continue
                        
                    dist = math.sqrt(sum((a-b)**2 for a, b in zip(node.vector_position, pos)))
                    if dist < 5:
                        # Extract the main word from context or next_words
                        if node.context:
                            concept_words.append(node.context[-1])
            
            # Get most common words in this area
            common_words = [w for w, _ in Counter(concept_words).most_common(5)]
            
            concepts.append({
                'position': [int(c) for c in pos],
                'density': float(density),
                'words': common_words
            })
            
        return concepts
        
    def save_model(self, model_name: str = None):
        """Save complete model state with training history"""
        if model_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"voxelpath_model_{timestamp}"
        save_path = os.path.join(self.model_dir, model_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save N-Gram memory
        ngram_path = os.path.join(save_path, "ngram_memory.pkl.gz")
        with gzip.open(ngram_path, 'wb') as f:
            # Convert defaultdict to dict for serialization
            serializable_ngrams = {}
            for level, nodes in self.ngram_memory.items():
                serializable_ngrams[level] = dict(nodes)
            pickle.dump(serializable_ngrams, f)
            
        # Save voxel tensor
        voxel_path = os.path.join(save_path, "voxel_tensor.npy")
        np.save(voxel_path, self.voxel_tensor)
        
        # Save spatial graph
        graph_path = os.path.join(save_path, "spatial_graph.pkl.gz")
        with gzip.open(graph_path, 'wb') as f:
            serializable_graph = {k: list(v) for k, v in self.spatial_graph.items()}
            pickle.dump(serializable_graph, f)
            
        # Save trajectory tunnels
        tunnel_path = os.path.join(save_path, "trajectory_tunnels.pkl.gz")
        with gzip.open(tunnel_path, 'wb') as f:
            pickle.dump(self.trajectory_tunnels, f)
            
        # Save training history
        history_path = os.path.join(save_path, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump({
                'training_history': self.training_history,
                'validation_history': self.validation_history,
                'total_training_time': self.training_time,
                'tokens_processed': self.tokens_processed
            }, f, indent=2)
            
        # Save metadata
        metadata = {
            'tokens_processed': self.tokens_processed,
            'ngrams_learned': {str(k): v for k, v in self.ngrams_learned.items()},
            'resolution': self.resolution,
            'save_time': datetime.datetime.now().isoformat(),
            'memory_usage_mb': self.memory_usage_mb,
            'training_time': self.training_time,
            'model_version': '3.1'
        }
        meta_path = os.path.join(save_path, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"[SAVE] Model saved to {save_path}")
        return save_path
        
    def load_model(self, model_path: str):
        """Load model from saved state with version compatibility"""
        print(f"[LOAD] Loading model from {model_path}")
        
        # Load metadata first
        meta_path = os.path.join(model_path, "metadata.json")
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print("[LOAD] No metadata found, using defaults")
            metadata = {
                'tokens_processed': 0,
                'ngrams_learned': {n: 0 for n in range(2, 6)},
                'resolution': self.resolution,
                'memory_usage_mb': 0,
                'training_time': 0
                }
        
        # Restore metadata
        self.tokens_processed = metadata.get('tokens_processed', 0)
        ngrams_learned_raw = metadata.get('ngrams_learned', {})
        self.ngrams_learned = {int(k): v for k, v in ngrams_learned_raw.items()}
        self.training_time = metadata.get('training_time', 0)
        
        # Recalculate if empty/zero (handling legacy/broken saves)
        if sum(self.ngrams_learned.values()) == 0:
             self.ngrams_learned = {n: 0 for n in range(2, 6)} # Reset default
        
        print(f"[LOAD] Metadata restored. Tokens: {self.tokens_processed}")
        ngram_path = os.path.join(model_path, "ngram_memory.pkl.gz")
        try:
            with gzip.open(ngram_path, 'rb') as f:
                loaded_ngrams = pickle.load(f)
                
            # Convert back to defaultdict with proper types
            self.ngram_memory.clear()
            for key, nodes in loaded_ngrams.items():
                # Handle both Enum and integer keys from pickle/json
                level = None
                if isinstance(key, NGramLevel):
                    level = key
                elif isinstance(key, int):
                     try:
                        level = NGramLevel(key)
                     except ValueError:
                        continue
                elif isinstance(key, str):
                     # Handle "NGramLevel.BIGRAM" or just "2"
                     if key.isdigit():
                         try:
                            level = NGramLevel(int(key))
                         except ValueError:
                            continue
                
                if level:
                    self.ngram_memory[level] = defaultdict(
                        lambda: NGramNode(context=()), 
                        nodes
                    )
            
            # Simple validation
            total_nodes = sum(len(n) for n in self.ngram_memory.values())
            print(f"[LOAD] Restored {total_nodes} N-Gram nodes across {len(self.ngram_memory)} levels")

            # [OPTIMIZATION] Compute global unigram frequencies for reliable fallback
            print("[LOAD] Computing fallback statistics...")
            self.fallback_unigrams = {}
            if NGramLevel.BIGRAM in self.ngram_memory:
                raw_counts = defaultdict(float)
                for node in self.ngram_memory[NGramLevel.BIGRAM].values():
                     for next_word, freq in node.next_words.items():
                         raw_counts[next_word] += freq
                
                # Normalize
                total_uni = sum(raw_counts.values())
                if total_uni > 0:
                    self.fallback_unigrams = {w: c/total_uni for w, c in raw_counts.items()}
            print(f"[LOAD] Fallback stats computed: {len(self.fallback_unigrams)} unique tokens")

            
        except Exception as e:
            print(f"[LOAD] Error loading N-Grams: {e}")
            self.ngram_memory = {
                NGramLevel.BIGRAM: defaultdict(lambda: NGramNode(context=())),
                NGramLevel.TRIGRAM: defaultdict(lambda: NGramNode(context=())),
                NGramLevel.QUADGRAM: defaultdict(lambda: NGramNode(context=())),
                NGramLevel.QUINTGRAM: defaultdict(lambda: NGramNode(context=())),
            }
            
        # Load voxel tensor
        voxel_path = os.path.join(model_path, "voxel_tensor.npy")
        try:
            self.voxel_tensor = np.load(voxel_path)
            # Ensure correct shape
            if self.voxel_tensor.shape != (self.resolution, self.resolution, self.resolution):
                print("[LOAD] Voxel tensor shape mismatch, creating new one")
                self.voxel_tensor = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float32)
        except Exception as e:
            print(f"[LOAD] Error loading voxel tensor: {e}")
            self.voxel_tensor = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float32)
            
        # Load spatial graph
        graph_path = os.path.join(model_path, "spatial_graph.pkl.gz")
        try:
            with gzip.open(graph_path, 'rb') as f:
                loaded_graph = pickle.load(f)
            self.spatial_graph.clear()
            for pos_str, neighbors in loaded_graph.items():
                # Handle position format
                if isinstance(pos_str, str):
                    # Parse string representation of tuple
                    pos = eval(pos_str)
                elif isinstance(pos_str, (list, tuple)):
                    pos = tuple(int(x) for x in pos_str[:3])
                else:
                    continue
                    
                self.spatial_graph[pos] = set(
                    tuple(int(x) for x in n[:3]) if isinstance(n, (list, tuple)) else eval(str(n))
                    for n in neighbors
                )
        except Exception as e:
            print(f"[LOAD] Error loading spatial graph: {e}")
            self.spatial_graph = defaultdict(set)
            
        # Load trajectory tunnels
        tunnel_path = os.path.join(model_path, "trajectory_tunnels.pkl.gz")
        try:
            with gzip.open(tunnel_path, 'rb') as f:
                self.trajectory_tunnels = pickle.load(f)
        except Exception as e:
            print(f"[LOAD] Error loading trajectory tunnels: {e}")
            self.trajectory_tunnels = {}
            
        # Load training history if available
        history_path = os.path.join(model_path, "training_history.json")
        try:
            with open(history_path, 'r') as f:
                history_data = json.load(f)
                self.training_history = history_data.get('training_history', [])
                self.validation_history = history_data.get('validation_history', [])
                self.training_time = history_data.get('total_training_time', 0)
                self.tokens_processed = history_data.get('tokens_processed', 0)
        except Exception as e:
            print(f"[LOAD] Error loading training history: {e}")
            self.training_history = []
            self.validation_history = []
            
        # Restore metadata
        self.tokens_processed = metadata.get('tokens_processed', self.tokens_processed)
        self.ngrams_learned = {int(k): v for k, v in metadata.get('ngrams_learned', {str(n):0 for n in range(2,6)}).items()}
        
        # If metadata is missing or zero (migration/bug fix), recalculate from memory
        total_learned = sum(self.ngrams_learned.values())
        if total_learned == 0 and self.ngram_memory:
            print("[LOAD] Recalculating N-Gram statistics from memory...")
            for level_enum, nodes in self.ngram_memory.items():
                self.ngrams_learned[level_enum.value] = len(nodes)
        
        self.memory_usage_mb = metadata.get('memory_usage_mb', 0)
        self.training_time = metadata.get('training_time', self.training_time)
        
        # Update memory stats
        self._update_memory_stats()
        
        print(f"[LOAD] Model loaded successfully. {self.tokens_processed:,} tokens processed, "
              f"{sum(self.ngrams_learned.values()):,} N-Grams learned")
        return True
        
    def compare_models(self, model_paths: List[str]) -> Dict[str, Dict]:
        """Compare multiple models and return comprehensive statistics"""
        results = {}
        for model_path in model_paths:
            try:
                # Load metadata
                meta_path = os.path.join(model_path, "metadata.json")
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Load training history
                history_path = os.path.join(model_path, "training_history.json")
                training_history = []
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        history_data = json.load(f)
                        training_history = history_data.get('training_history', [])
                        
                model_name = os.path.basename(model_path)
                results[model_name] = {
                    "tokens_processed": metadata.get('tokens_processed', 0),
                    "ngrams_learned": metadata.get('ngrams_learned', {}),
                    "memory_usage": metadata.get('memory_usage_mb', 0),
                    "training_time": metadata.get('training_time', 0),
                    "save_time": metadata.get('save_time', ''),
                    "training_history": training_history,
                    "final_perplexity": float('inf'),
                    "best_perplexity": float('inf')
                }
                
                # Get best and final perplexity from history
                if training_history:
                    perplexities = [entry.get('perplexity', float('inf')) for entry in training_history if entry.get('perplexity') is not None]
                    if perplexities:
                        results[model_name]["final_perplexity"] = perplexities[-1]
                        results[model_name]["best_perplexity"] = min(perplexities)
                        
            except Exception as e:
                model_name = os.path.basename(model_path)
                results[model_name] = {
                    "error": str(e)
                }
                
        return results
    
    def _get_fallback_probabilities(self) -> Dict[str, float]:
        """Get fallback probabilities from global unigram stats"""
        if hasattr(self, 'fallback_unigrams') and self.fallback_unigrams:
            return self.fallback_unigrams.copy()
            
        # Fallback to sampling bigrams if unigrams not computed
        fallback = {}
        if NGramLevel.BIGRAM in self.ngram_memory:
            bigrams = self.ngram_memory[NGramLevel.BIGRAM]
            # Sample first 20 nodes
            count = 0
            for node in bigrams.values():
                for next_word, freq in node.next_words.items():
                    fallback[next_word] = fallback.get(next_word, 0) + freq
                    count += 1
                if count > 200: # Increased sample size
                    break
        
        # Normalize
        total = sum(fallback.values())
        if total > 0:
            return {w: c/total for w, c in fallback.items()}
        return {}

    def sync_to_storage(self, progress_callback=None):
        """
        Sync in-memory N-grams to persistent SQLite storage using BATCH operations.
        """
        print("[SYNC] Starting synchronization to SQLite...")
        total_nodes = sum(len(n) for n in self.ngram_memory.values())
        processed = 0
        batch_size = 5000
        transitions_batch = []
        
        try:
            # We iterate through all levels
            for level, nodes in self.ngram_memory.items():
                for context_tuple, node in nodes.items():
                    
                    # Prepare batch for storage
                    for next_word, freq in node.next_words.items():
                        transitions_batch.append((context_tuple, next_word, freq))
                    
                    processed += 1
                    
                    # Flush batch
                    if len(transitions_batch) >= batch_size:
                        self.storage.batch_add_transitions(transitions_batch)
                        transitions_batch = []
                        
                        if progress_callback:
                            progress = (processed / total_nodes) * 100
                            progress_callback(progress, f"Syncing {processed}/{total_nodes} nodes...")
            
            # Flush remaining
            if transitions_batch:
                self.storage.batch_add_transitions(transitions_batch)
                
            print("[SYNC] Complete.")
        except Exception as e:
            print(f"[SYNC] Error during sync: {e}")
            import traceback
            traceback.print_exc() 
