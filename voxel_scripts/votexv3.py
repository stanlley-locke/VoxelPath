"""
VoxelPath Production System v3.0
A Geometric 3D Coordinate-Mapped Trajectory Model with Hierarchical N-Gram Memory
and Stochastic Graph-Based Architecture for Shakespearean Text Generation
"""
import sys
import numpy as np
import json
import os
import random
import datetime
import re
import math
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import threading
import concurrent.futures
import pickle
import gzip
from enum import Enum
import time
import webbrowser
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QTextEdit, QLabel, QProgressBar,
    QFileDialog, QFrame, QSplitter, QSlider, QMessageBox,
    QGroupBox, QSpinBox, QCheckBox, QTabWidget, QListWidget,
    QListWidgetItem, QComboBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QLineEdit, QToolBar, QStatusBar,
    QMenu, QMenuBar, QInputDialog, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize, QUrl
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QAction, QDesktopServices
import qdarktheme
# Visualization Imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================
class NGramLevel(Enum):
    """Hierarchical N-Gram memory levels"""
    BIGRAM = 2
    TRIGRAM = 3
    QUADGRAM = 4
    QUINTGRAM = 5
    
@dataclass
class NGramNode:
    """Weighted directed graph node for N-Gram context"""
    context: Tuple[str, ...]
    next_words: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    total_count: float = 0.0
    vector_position: Optional[Tuple[int, int, int]] = None
    
    def add_transition(self, word: str, weight: float = 1.0):
        """Add a weighted transition from this context"""
        self.next_words[word] = self.next_words.get(word, 0.0) + weight
        self.total_count += weight
        
    def get_probability(self, word: str) -> float:
        """Get normalized probability for a transition"""
        if self.total_count == 0:
            return 0.0
        return self.next_words.get(word, 0.0) / self.total_count
        
    def get_logits(self, temperature: float = 1.0) -> Dict[str, float]:
        """Get temperature-scaled logits for all possible next words"""
        if not self.next_words:
            return {}
        # Apply temperature scaling
        if temperature != 1.0:
            logits = {w: np.power(score, 1.0 / temperature) 
                     for w, score in self.next_words.items()}
        else:
            logits = dict(self.next_words)
        # Normalize
        total = sum(logits.values())
        if total > 0:
            return {w: s / total for w, s in logits.items()}
        return logits
        
@dataclass
class TrajectoryTunnel:
    """Fast path for frequently occurring patterns"""
    pattern: str
    continuation: str
    usage_count: int = 0
    last_used: Optional[datetime.datetime] = None
    confidence: float = 0.0
    
    def should_activate(self, min_confidence: float = 0.8) -> bool:
        """Check if tunnel should be used based on confidence"""
        return self.confidence >= min_confidence
        
@dataclass
class GeometricMomentum:
    """Track movement through voxel space"""
    position: Tuple[int, int, int]
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    def update(self, new_position: Tuple[int, int, int], alpha: float = 0.1):
        """Update momentum with exponential smoothing"""
        dx = new_position[0] - self.position[0]
        dy = new_position[1] - self.position[1]
        dz = new_position[2] - self.position[2]
        
        # Update velocity with smoothing
        self.velocity = (
            (1 - alpha) * self.velocity[0] + alpha * dx,
            (1 - alpha) * self.velocity[1] + alpha * dy,
            (1 - alpha) * self.velocity[2] + alpha * dz
        )
        
        # Update acceleration (second derivative)
        old_velocity = self.velocity
        self.acceleration = (
            self.velocity[0] - old_velocity[0],
            self.velocity[1] - old_velocity[1],
            self.velocity[2] - old_velocity[2]
        )
        
        self.position = new_position
        
@dataclass
class TextAnalysisMetrics:
    """Metrics for analyzing generated text quality"""
    perplexity: float = 0.0
    coherence_score: float = 0.0
    diversity_score: float = 0.0
    repetition_rate: float = 0.0
    avg_word_length: float = 0.0
    sentence_complexity: float = 0.0
    
# ============================================================================
# TOKENIZATION ENGINE
# ============================================================================
class IntelligentTokenizer:
    """Advanced tokenizer with punctuation handling and ASCII sanitization"""
    # Regex patterns for sophisticated tokenization
    PUNCTUATION_PATTERN = r'([\.,!?;:"\'\-–—()\[\]{}])'
    WORD_PATTERN = r'(\w+)'
    NUMBER_PATTERN = r'(\d+)'
    
    # ASCII safe mapping for non-standard characters
    ASCII_SAFE_MAP = {
        '—': '-',  # Em dash to hyphen
        '–': '-',  # En dash to hyphen
        '“': '"',  # Smart quotes
        '”': '"',
        '‘': "'",
        '’': "'",
        '…': '...',  # Ellipsis
        '«': '"',
        '»': '"'
    }
    
    def __init__(self):
        # Combined pattern that captures words, numbers, and punctuation
        self.token_pattern = re.compile(
            f'{self.WORD_PATTERN}|{self.NUMBER_PATTERN}|{self.PUNCTUATION_PATTERN}'
        )
        # Define punctuation sets for detokenization
        self._punctuation_no_space_before = set('.,!?;:"\'') | set(')-]}>')
        self._punctuation_no_space_after = set('([{<')
        
    def sanitize_text(self, text: str) -> str:
        """Convert non-standard characters to ASCII-safe equivalents"""
        for old, new in self.ASCII_SAFE_MAP.items():
            text = text.replace(old, new)
        # Remove any remaining non-ASCII characters
        text = ''.join(char for char in text if ord(char) < 128)
        return text
        
    def tokenize(self, text: str) -> List[str]:
        """Intelligent tokenization preserving punctuation as separate tokens"""
        text = self.sanitize_text(text)
        tokens = []
        # Use regex to split while keeping delimiters
        matches = self.token_pattern.finditer(text)
        for match in matches:
            token = match.group()
            if token:  # Filter out empty matches
                tokens.append(token.lower())  # Case normalization
        return tokens
        
    def detokenize(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens with proper spacing"""
        result = []
        for i, token in enumerate(tokens):
            if i == 0:
                result.append(token)
            elif token in self._punctuation_no_space_before:
                # Attach punctuation without space
                result[-1] = result[-1] + token
            elif result and result[-1][-1] in self._punctuation_no_space_after:
                # Attach to opening punctuation without space
                result[-1] = result[-1] + token
            else:
                # Add space before word/number
                result.append(' ' + token)
                
        # Capitalize first letter
        if result:
            result[0] = result[0].capitalize()
        return ''.join(result)
        
    def analyze_text_metrics(self, tokens: List[str]) -> TextAnalysisMetrics:
        """Analyze text quality metrics"""
        metrics = TextAnalysisMetrics()
        
        # Calculate average word length
        word_lengths = [len(token) for token in tokens if token.isalpha()]
        if word_lengths:
            metrics.avg_word_length = sum(word_lengths) / len(word_lengths)
            
        # Calculate repetition rate
        unique_tokens = set(tokens)
        if tokens:
            metrics.repetition_rate = 1.0 - (len(unique_tokens) / len(tokens))
            
        # Calculate sentence complexity (rough estimate)
        sentence_delimiters = ['.', '!', '?', '\n']
        sentences = []
        current_sentence = []
        for token in tokens:
            current_sentence.append(token)
            if token in sentence_delimiters:
                sentences.append(current_sentence)
                current_sentence = []
        if current_sentence:
            sentences.append(current_sentence)
            
        if sentences:
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
            metrics.sentence_complexity = avg_sentence_length
            
        return metrics
        
# ============================================================================
# CORE INTELLIGENCE ENGINE
# ============================================================================
class VoxelPathProductionEngine:
    """Production-grade stochastic graph-based architecture"""
    def __init__(self, voxel_resolution: int = 128, model_dir: str = "voxel_models"):
        # Configuration
        self.resolution = voxel_resolution
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Core data structures
        self.tokenizer = IntelligentTokenizer()
        self.reset_engine()
        
        # Training statistics
        self.tokens_processed = 0
        self.ngrams_learned = {n: 0 for n in range(2, 6)}
        self.training_time = 0.0
        
        # System performance
        self.memory_usage_mb = 0
        self.last_optimization = datetime.datetime.now()
        self.start_time = time.time()
        
    def reset_engine(self):
        """Complete engine reset - fresh start"""
        print("[ENGINE] Resetting engine to initial state")
        # Hierarchical N-Gram memory
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
        self.geometric_momentum = GeometricMomentum(position=(0, 0, 0))
        self.recent_context = deque(maxlen=15)  # For repetition penalty
        
        # Training state
        self.is_training = False
        self.current_corpus = None
        self.optimization_scheduled = False
        print("[ENGINE] Engine reset complete")
        
    def _calculate_3d_position(self, token: str) -> Tuple[int, int, int]:
        """Map token to 3D coordinate using intelligent hashing"""
        # Use first three characters or pad
        padded = (token + '   ')[:3]
        # Convert to ASCII values
        x = ord(padded[0]) if ord(padded[0]) < self.resolution else 0
        y = ord(padded[1]) if ord(padded[1]) < self.resolution else 0
        z = ord(padded[2]) if ord(padded[2]) < self.resolution else 0
        return (x, y, z)
        
    def train_corpus(self, text: str, batch_size: int = 10000):
        """Advanced training with hierarchical N-Gram learning"""
        start_time = time.time()
        print(f"[TRAIN] Starting training on {len(text):,} characters")
        # Tokenize entire corpus
        tokens = self.tokenizer.tokenize(text)
        self.tokens_processed += len(tokens)
        
        # Process in batches for memory efficiency
        for batch_start in range(0, len(tokens), batch_size):
            batch_tokens = tokens[batch_start:batch_start + batch_size]
            # Train N-Grams from 2 to 5
            for n in range(2, 6):
                for i in range(len(batch_tokens) - n + 1):
                    context = tuple(batch_tokens[i:i+n-1])
                    next_word = batch_tokens[i+n-1]
                    # Get appropriate N-Gram level
                    ngram_level = NGramLevel(n)
                    node_key = context
                    # Update N-Gram node
                    node = self.ngram_memory[ngram_level][node_key]
                    node.add_transition(next_word, weight=1.0)
                    # Update context
                    node.context = context
                    # Map to 3D space
                    pos = self._calculate_3d_position(next_word)
                    node.vector_position = pos
                    # Update voxel density
                    self.voxel_tensor[pos] += 1.0
                    
                    # Update spatial graph
                    if node.vector_position:
                        if len(context) > 0:
                            prev_context = context[:-1] if len(context) > 1 else ()
                            prev_node = self.ngram_memory[ngram_level].get(prev_context)
                            if prev_node and prev_node.vector_position:
                                self.spatial_graph[prev_node.vector_position].add(
                                    node.vector_position
                                )
            # Update statistics
            for n in range(2, 6):
                self.ngrams_learned[n] = len(self.ngram_memory[NGramLevel(n)])
                
            # Periodic optimization
            if batch_start % 100000 == 0:
                self._optimize_memory()
                print(f"[TRAIN] Processed {batch_start:,} tokens")
                
        self.training_time += time.time() - start_time
        print(f"[TRAIN] Training complete. Learned {sum(self.ngrams_learned.values()):,} N-Grams")
        
    def recursive_backoff(self, context: Tuple[str, ...]) -> Dict[str, float]:
        """
        Intelligent backoff from 5-gram down to 2-gram
        Returns weighted combination of probabilities from all levels
        """
        probabilities = defaultdict(float)
        total_weight = 0.0
        
        # Try from 5-gram down to 2-gram
        for n in range(5, 1, -1):
            if len(context) >= n - 1:
                ngram_level = NGramLevel(n)
                context_slice = context[-(n-1):] if n > 1 else ()
                node = self.ngram_memory[ngram_level].get(context_slice)
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
        # Calculate penalty for each word
        for word in list(penalized_probs.keys()):
            # Count occurrences in recent context
            recent_count = recent_tokens.count(word)
            if recent_count > 0:
                # Apply exponential penalty
                penalty = penalty_strength ** recent_count
                penalized_probs[word] /= penalty
                
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
        A* pathfinding through voxel space
        """
        if target_chars is None:
            target_chars = [' ', '.', '!', '?', '\n']
        target_positions = [self._calculate_3d_position(c) for c in target_chars]
        
        # Priority queue: (f_score, g_score, position, path)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_pos, []))
        g_scores = {start_pos: 0}
        visited = set()
        iterations = 0
        
        while open_set and iterations < max_iterations:
            f_score, g_score, current, path = heapq.heappop(open_set)
            iterations += 1
            
            # Check if reached target
            if current in target_positions:
                return path + [current]
                
            visited.add(current)
            
            # Explore neighbors
            for neighbor in self.spatial_graph.get(current, []):
                if neighbor in visited:
                    continue
                    
                # Cost based on voxel density (higher density = lower cost)
                if 0 <= neighbor[0] < self.resolution and \
                   0 <= neighbor[1] < self.resolution and \
                   0 <= neighbor[2] < self.resolution:
                    density = self.voxel_tensor[neighbor]
                else:
                    density = 1.0
                    
                tentative_g = g_score + (1.0 / (density + 1e-6))
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    # Heuristic: Euclidean distance to nearest target
                    distances = [math.sqrt((neighbor[0]-t[0])**2 + 
                                         (neighbor[1]-t[1])**2 + 
                                         (neighbor[2]-t[2])**2)
                                for t in target_positions]
                    h_score = min(distances) if distances else 0
                    f_score_new = tentative_g + h_score
                    heapq.heappush(open_set,
                                 (f_score_new, tentative_g, neighbor,
                                 path + [current]))
                                 
        # Return best path found
        return path + [current] if path else [start_pos]
        
    def generate_text(self,
                     seed_text: str,
                     max_length: int = 200,
                     temperature: float = 0.85,
                     top_p: float = 0.9,
                     repetition_penalty: float = 1.2,
                     use_astar: bool = False,
                     use_momentum: bool = True) -> Tuple[str, List[Tuple[int, int, int]], TextAnalysisMetrics]:
        """
        Advanced text generation with all features enabled
        """
        # Tokenize seed
        seed_tokens = self.tokenizer.tokenize(seed_text)
        if len(seed_tokens) < 1:
            # Default to Shakespearean start
            seed_tokens = ['to', 'be', 'or', 'not', 'to', 'be']
            
        generated_tokens = seed_tokens.copy()
        trajectory = []
        metrics = TextAnalysisMetrics()
        
        # Track recent context for repetition penalty
        recent_context = deque(maxlen=20)
        recent_context.extend(seed_tokens)
        
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
                    continue
                    
            # Get probabilities with recursive backoff
            probabilities = self.recursive_backoff(context)
            if not probabilities:
                # Fallback to simple sampling
                possible_nodes = list(self.ngram_memory[NGramLevel.BIGRAM].values())
                if possible_nodes:
                    # Get the most frequent word from a random node
                    node = random.choice(possible_nodes)
                    if node.next_words:
                        next_word = max(node.next_words.items(), key=lambda x: x[1])[0]
                    else:
                        next_word = 'the'
                else:
                    next_word = 'the'
                    
                generated_tokens.append(next_word)
                recent_context.append(next_word)
                # Update trajectory
                pos = self._calculate_3d_position(next_word)
                trajectory.append(pos)
                continue
                
            # Apply repetition penalty
            probabilities = self.apply_repetition_penalty(
                probabilities,
                list(recent_context)[-10:],  # Last 10 tokens
                repetition_penalty
            )
            
            # Apply temperature
            if temperature != 1.0:
                # Softmax with temperature
                logits = np.array(list(probabilities.values()))
                scaled_logits = logits / temperature
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                probabilities = dict(zip(probabilities.keys(), exp_logits / exp_logits.sum()))
                
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_probs = sorted(probabilities.items(), key=lambda x: -x[1])
                cumulative = 0.0
                cutoff_index = len(sorted_probs)
                for i, (word, prob) in enumerate(sorted_probs):
                    cumulative += prob
                    if cumulative >= top_p:
                        cutoff_index = i + 1
                        break
                        
                # Keep only top-p words
                kept_words = {word: prob for word, prob in sorted_probs[:cutoff_index]}
                # Renormalize
                total = sum(kept_words.values())
                if total > 0:
                    probabilities = {w: p / total for w, p in kept_words.items()}
                else:
                    probabilities = dict(sorted_probs[:cutoff_index])
                    
            # Sample from distribution
            if probabilities:
                words, probs = zip(*probabilities.items())
                try:
                    next_word = random.choices(words, weights=probs, k=1)[0]
                except ValueError:
                    # Fallback if weights are invalid
                    next_word = random.choice(words)
            else:
                # Fallback
                next_word = 'the'
                
            # Update
            generated_tokens.append(next_word)
            recent_context.append(next_word)
            
            # Update trajectory
            pos = self._calculate_3d_position(next_word)
            trajectory.append(pos)
            
            # Update geometric momentum if enabled
            if use_momentum and len(trajectory) > 1:
                self.geometric_momentum.update(pos)
                
        # Convert tokens back to text
        generated_text = self.tokenizer.detokenize(generated_tokens)
        # Analyze text metrics
        metrics = self.tokenizer.analyze_text_metrics(generated_tokens)
        
        return generated_text, trajectory, metrics
        
    def _optimize_memory(self):
        """Memory optimization and cleanup"""
        print("[OPTIMIZE] Running memory optimization...")
        # Prune low-frequency N-Grams
        for level_name, level in self.ngram_memory.items():
            to_delete = []
            for key, node in level.items():
                if node.total_count < 2:  # Keep only N-Grams seen at least twice
                    to_delete.append(key)
            for key in to_delete:
                del level[key]
                
        # Clear spatial graph for pruned nodes
        self.spatial_graph.clear()
        
        # Rebuild spatial graph with remaining nodes
        for level_name, nodes in self.ngram_memory.items():
            for node in nodes.values():
                if node.vector_position:
                    # Find connections to other nodes
                    for other_node in nodes.values():
                        if other_node.vector_position and other_node != node:
                            # Simple distance-based connection
                            try:
                                dist = math.sqrt(
                                    (node.vector_position[0] - other_node.vector_position[0])**2 +
                                    (node.vector_position[1] - other_node.vector_position[1])**2 +
                                    (node.vector_position[2] - other_node.vector_position[2])**2
                                )
                                if dist < 50:  # Connection threshold
                                    self.spatial_graph[node.vector_position].add(
                                        other_node.vector_position
                                    )
                            except Exception as e:
                                continue
                                
        # Update memory usage estimate
        self._update_memory_stats()
        self.last_optimization = datetime.datetime.now()
        print(f"[OPTIMIZE] Memory optimized. Current usage: {self.memory_usage_mb:.1f} MB")
        
    def _update_memory_stats(self):
        """Calculate current memory usage"""
        # Estimate N-Gram memory
        ngram_memory = 0
        for level, nodes in self.ngram_memory.items():
            for node in nodes.values():
                ngram_memory += len(str(node.context)) * 2  # Unicode overhead
                ngram_memory += sum(len(str(w)) * 2 for w in node.next_words.keys())
                
        # Voxel tensor memory (float32)
        voxel_memory = self.voxel_tensor.nbytes
        
        # Graph memory
        graph_memory = 0
        for pos, neighbors in self.spatial_graph.items():
            graph_memory += 24  # Tuple overhead
            graph_memory += len(neighbors) * 24
            
        total_bytes = ngram_memory + voxel_memory + graph_memory
        self.memory_usage_mb = total_bytes / (1024 * 1024)
        
    def save_model(self, model_name: str = None):
        """Save complete model state"""
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
            
        # Save metadata
        metadata = {
            'tokens_processed': self.tokens_processed,
            'ngrams_learned': self.ngrams_learned,
            'resolution': self.resolution,
            'save_time': datetime.datetime.now().isoformat(),
            'memory_usage_mb': self.memory_usage_mb,
            'training_time': self.training_time
        }
        meta_path = os.path.join(save_path, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"[SAVE] Model saved to {save_path}")
        return save_path
        
    def load_model(self, model_path: str):
        """Load model from saved state"""
        print(f"[LOAD] Loading model from {model_path}")
        # Load metadata
        meta_path = os.path.join(model_path, "metadata.json")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        # Load N-Gram memory
        ngram_path = os.path.join(model_path, "ngram_memory.pkl.gz")
        with gzip.open(ngram_path, 'rb') as f:
            loaded_ngrams = pickle.load(f)
            
        # Convert back to defaultdict
        self.ngram_memory.clear()
        for level, nodes in loaded_ngrams.items():
            level_enum = NGramLevel(level.value)
            self.ngram_memory[level_enum] = defaultdict(lambda: NGramNode(context=()), nodes)
            
        # Load voxel tensor
        voxel_path = os.path.join(model_path, "voxel_tensor.npy")
        self.voxel_tensor = np.load(voxel_path)
        
        # Load spatial graph
        graph_path = os.path.join(model_path, "spatial_graph.pkl.gz")
        with gzip.open(graph_path, 'rb') as f:
            loaded_graph = pickle.load(f)
        self.spatial_graph.clear()
        for pos, neighbors in loaded_graph.items():
            self.spatial_graph[tuple(pos)] = set(tuple(n) for n in neighbors)
            
        # Restore metadata
        self.tokens_processed = metadata['tokens_processed']
        self.ngrams_learned = metadata['ngrams_learned']
        self.memory_usage_mb = metadata.get('memory_usage_mb', 0)
        self.training_time = metadata.get('training_time', 0)
        
        print(f"[LOAD] Model loaded successfully. {self.tokens_processed:,} tokens processed")
        return True
        
    def compare_models(self, model_paths: List[str]) -> Dict[str, Dict]:
        """Compare multiple models and return statistics"""
        results = {}
        for model_path in model_paths:
            try:
                # Load metadata
                meta_path = os.path.join(model_path, "metadata.json")
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    
                model_name = os.path.basename(model_path)
                results[model_name] = {
                    "tokens_processed": metadata.get('tokens_processed', 0),
                    "ngrams_learned": metadata.get('ngrams_learned', {}),
                    "memory_usage": metadata.get('memory_usage_mb', 0),
                    "training_time": metadata.get('training_time', 0),
                    "save_time": metadata.get('save_time', '')
                }
            except Exception as e:
                model_name = os.path.basename(model_path)
                results[model_name] = {
                    "error": str(e)
                }
                
        return results
        
# ============================================================================
# ASYNCHRONOUS WORKERS
# ============================================================================
class TrainingWorker(QThread):
    """Advanced training worker with real-time progress reporting"""
    # Signals
    progress_updated = pyqtSignal(int, str)  # (progress_percent, status_message)
    training_complete = pyqtSignal(str)  # completion_message
    error_occurred = pyqtSignal(str)  # error_message
    stats_updated = pyqtSignal(dict)  # training statistics
    
    def __init__(self, engine: VoxelPathProductionEngine, corpus_path: str):
        super().__init__()
        self.engine = engine
        self.corpus_path = corpus_path
        self.is_running = True
        
    def run(self):
        """Main training execution"""
        try:
            # Read corpus
            self.progress_updated.emit(0, f"Reading corpus: {os.path.basename(self.corpus_path)}")
            with open(self.corpus_path, 'r', encoding='utf-8', errors='replace') as f:
                corpus_text = f.read()
            if len(corpus_text) > 50 * 1024 * 1024:  # 50MB limit
                self.error_occurred.emit("Corpus too large (max 50MB)")
                return
                
            # Estimate chunks for progress reporting
            total_chars = len(corpus_text)
            chunk_size = 100000
            total_chunks = (total_chars // chunk_size) + 1
            
            self.progress_updated.emit(5, f"Processing {total_chars:,} characters...")
            
            # Process in chunks
            for chunk_idx in range(0, total_chars, chunk_size):
                if not self.is_running:
                    break
                    
                chunk = corpus_text[chunk_idx:chunk_idx + chunk_size]
                self.engine.train_corpus(chunk, batch_size=20000)
                
                # Update progress
                progress = min(95, int((chunk_idx / total_chars) * 90) + 5)
                status = f"Processed {chunk_idx:,} of {total_chars:,} characters"
                self.progress_updated.emit(progress, status)
                
                # Update statistics periodically
                if chunk_idx % 500000 == 0:
                    stats = {
                        'tokens_processed': self.engine.tokens_processed,
                        'ngrams_learned': self.engine.ngrams_learned.copy(),
                        'memory_usage_mb': self.engine.memory_usage_mb
                    }
                    self.stats_updated.emit(stats)
                    
            if self.is_running:
                # Final optimization
                self.progress_updated.emit(96, "Optimizing memory...")
                self.engine._optimize_memory()
                
                # Save model
                self.progress_updated.emit(98, "Saving model snapshot...")
                save_path = self.engine.save_model()
                self.progress_updated.emit(100, "Training complete!")
                self.training_complete.emit(f"Model saved to {save_path}")
        except Exception as e:
            self.error_occurred.emit(f"Training error: {str(e)}")
            
    def stop(self):
        """Graceful stop"""
        self.is_running = False
        
class GenerationWorker(QThread):
    """Background text generation worker"""
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
                use_momentum: bool = True):
        super().__init__()
        self.engine = engine
        self.seed_text = seed_text
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.use_astar = use_astar
        self.use_momentum = use_momentum
        
    def run(self):
        """Execute generation"""
        try:
            self.generation_progress.emit(10, "Initializing generation...")
            # Generate text
            generated_text, trajectory, metrics = self.engine.generate_text(
                seed_text=self.seed_text,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                use_astar=self.use_astar,
                use_momentum=self.use_momentum
            )
            
            self.generation_progress.emit(100, "Generation complete!")
            
            # Convert metrics to dictionary for Qt signal
            metrics_dict = {
                'perplexity': metrics.perplexity,
                'coherence_score': metrics.coherence_score,
                'diversity_score': metrics.diversity_score,
                'repetition_rate': metrics.repetition_rate,
                'avg_word_length': metrics.avg_word_length,
                'sentence_complexity': metrics.sentence_complexity
            }
            
            self.generation_complete.emit(generated_text, trajectory, metrics_dict)
        except Exception as e:
            self.generation_complete.emit(f"Error: {str(e)}", [], {})
            
# ============================================================================
# ANALYSIS DIALOGS
# ============================================================================
class TextAnalysisDialog(QDialog):
    """Dialog for detailed text analysis"""
    def __init__(self, metrics: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Text Analysis Report")
        self.setMinimumSize(500, 400)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Text Quality Analysis")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Metrics table
        table = QTableWidget(6, 2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        metrics_data = [
            ("Perplexity", f"{metrics.get('perplexity', 0.0):.2f}"),
            ("Coherence Score", f"{metrics.get('coherence_score', 0.0):.2f}"),
            ("Diversity Score", f"{metrics.get('diversity_score', 0.0):.2f}"),
            ("Repetition Rate", f"{metrics.get('repetition_rate', 0.0):.2f}"),
            ("Avg Word Length", f"{metrics.get('avg_word_length', 0.0):.2f}"),
            ("Sentence Complexity", f"{metrics.get('sentence_complexity', 0.0):.2f}")
        ]
        
        for i, (metric, value) in enumerate(metrics_data):
            table.setItem(i, 0, QTableWidgetItem(metric))
            table.setItem(i, 1, QTableWidgetItem(value))
            
        layout.addWidget(table)
        
        # Interpretation
        interpretation = QLabel()
        interpretation.setWordWrap(True)
        interpretation.setStyleSheet("background-color: #2a2a3a; padding: 10px; border-radius: 5px;")
        
        rep_rate = metrics.get('repetition_rate', 0.0)
        diversity = metrics.get('diversity_score', 0.0)
        
        if rep_rate > 0.4:
            analysis = "The text shows high repetition, which may indicate model limitations. Consider increasing diversity parameters."
        elif diversity < 0.3:
            analysis = "Text diversity is low. Try increasing temperature or decreasing repetition penalty."
        else:
            analysis = "Text quality metrics are within acceptable ranges. Good balance of coherence and creativity."
            
        interpretation.setText(f"<b>Analysis:</b> {analysis}")
        layout.addWidget(interpretation)
        
        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
class BatchGenerationDialog(QDialog):
    """Dialog for batch text generation"""
    def __init__(self, engine: VoxelPathProductionEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setWindowTitle("Batch Text Generation")
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout()
        
        # Seed text
        seed_group = QGroupBox("Seed Text")
        seed_layout = QVBoxLayout()
        self.txt_seed = QTextEdit()
        self.txt_seed.setPlaceholderText("Enter seed text for batch generation")
        self.txt_seed.setMaximumHeight(100)
        seed_layout.addWidget(self.txt_seed)
        seed_group.setLayout(seed_layout)
        layout.addWidget(seed_group)
        
        # Parameters
        param_group = QGroupBox("Generation Parameters")
        param_layout = QVBoxLayout()
        
        # Number of texts
        num_layout = QHBoxLayout()
        num_layout.addWidget(QLabel("Number of texts:"))
        self.spin_num_texts = QSpinBox()
        self.spin_num_texts.setRange(1, 20)
        self.spin_num_texts.setValue(5)
        num_layout.addWidget(self.spin_num_texts)
        param_layout.addLayout(num_layout)
        
        # Length
        len_layout = QHBoxLayout()
        len_layout.addWidget(QLabel("Length per text:"))
        self.spin_text_length = QSpinBox()
        self.spin_text_length.setRange(50, 500)
        self.spin_text_length.setValue(150)
        len_layout.addWidget(self.spin_text_length)
        param_layout.addLayout(len_layout)
        
        # Temperature
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.slider_temp = QSlider(Qt.Orientation.Horizontal)
        self.slider_temp.setRange(10, 200)
        self.slider_temp.setValue(85)
        self.slider_temp.valueChanged.connect(self.update_temp_label)
        temp_layout.addWidget(self.slider_temp)
        self.lbl_temp_value = QLabel("0.85")
        temp_layout.addWidget(self.lbl_temp_value)
        param_layout.addLayout(temp_layout)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Output directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output directory:"))
        self.txt_output_dir = QLineEdit()
        self.txt_output_dir.setText(os.path.join(os.getcwd(), "generated_texts"))
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.txt_output_dir)
        dir_layout.addWidget(self.btn_browse)
        layout.addLayout(dir_layout)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.btn_generate = QPushButton("Generate Batch")
        self.btn_generate.clicked.connect(self.start_generation)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_generate)
        button_layout.addWidget(self.btn_cancel)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.generated_files = []
        
    def update_temp_label(self, value):
        self.lbl_temp_value.setText(f"{value/100:.2f}")
        
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.txt_output_dir.setText(directory)
            
    def start_generation(self):
        seed_text = self.txt_seed.toPlainText().strip()
        if not seed_text:
            QMessageBox.warning(self, "Input Error", "Please enter seed text.")
            return
            
        output_dir = self.txt_output_dir.text()
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Directory Error", f"Could not create directory: {str(e)}")
                return
                
        num_texts = self.spin_num_texts.value()
        text_length = self.spin_text_length.value()
        temperature = self.slider_temp.value() / 100
        
        self.btn_generate.setEnabled(False)
        self.progress.setValue(0)
        
        # Generate texts
        self.generated_files = []
        for i in range(num_texts):
            if self.isHidden():  # Dialog was closed
                break
                
            # Generate text
            generated_text, _, _ = self.engine.generate_text(
                seed_text=seed_text,
                max_length=text_length,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.2,
                use_astar=False,
                use_momentum=True
            )
            
            # Save to file
            filename = f"text_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}.txt"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            self.generated_files.append(filepath)
            
            # Update progress
            progress = int((i + 1) / num_texts * 100)
            self.progress.setValue(progress)
            QApplication.processEvents()
            
        self.btn_generate.setEnabled(True)
        QMessageBox.information(self, "Generation Complete", 
                              f"Generated {len(self.generated_files)} texts saved to:\n{output_dir}")
        self.accept()
        
# ============================================================================
# PRODUCTION GUI
# ============================================================================
class VoxelPathProductionUI(QMainWindow):
    """Industrial-grade production interface"""
    def __init__(self):
        super().__init__()
        # Set application style
        if QApplication.instance():
            QApplication.instance().setStyleSheet(qdarktheme.load_stylesheet("dark"))
            
        # Status bar - MUST be created before any logging happens
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System: Initializing...")
        
        # Core engine
        self.engine = VoxelPathProductionEngine()
        
        # Worker threads
        self.training_worker = None
        self.generation_worker = None
        
        # UI state
        self.current_trajectory = []
        self.loaded_models = []
        self.last_generated_text = ""
        
        # Setup UI
        self.setWindowTitle("VoxelPath Production System v3.0")
        self.resize(1600, 900)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create central widget and main layout
        self._setup_ui()
        
        # Initialize system
        self._system_log("System initialized. Ready for production.")
        
        # Status bar initialization
        self.status_bar.showMessage("System: Ready")
        
        # Update timer for real-time stats
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_status_bar)
        self.stats_timer.start(5000)  # Update every 5 seconds
        
    def _create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Engine", self)
        new_action.triggered.connect(self.reset_engine)
        file_menu.addAction(new_action)
        
        save_action = QAction("Save Model", self)
        save_action.triggered.connect(self.save_current_model)
        file_menu.addAction(save_action)
        
        load_action = QAction("Load Model", self)
        load_action.triggered.connect(self.load_model_dialog)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Generated Text", self)
        export_action.triggered.connect(self.export_generated_text)
        file_menu.addAction(export_action)
        
        batch_action = QAction("Batch Generation", self)
        batch_action.triggered.connect(self.open_batch_generation)
        file_menu.addAction(batch_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        optimize_action = QAction("Optimize Memory", self)
        optimize_action.triggered.connect(self.optimize_memory)
        tools_menu.addAction(optimize_action)
        
        compare_action = QAction("Compare Models", self)
        compare_action.triggered.connect(self.compare_models_dialog)
        tools_menu.addAction(compare_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self.open_documentation)
        help_menu.addAction(docs_action)
        
    def _setup_ui(self):
        """Setup complete production interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel: Control Center
        left_panel = self._create_control_panel()
        
        # Right Panel: Visualization & Output
        right_panel = self._create_visualization_panel()
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 1100])
        main_layout.addWidget(splitter)
        
    def _create_control_panel(self) -> QWidget:
        """Create the control panel with all configuration options"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # System Header
        header = QLabel("VoxelPath Control Center")
        header_font = QFont("Consolas", 12, QFont.Weight.Bold)
        header.setFont(header_font)
        header.setStyleSheet("color: #00ffff; padding: 10px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # System Console
        console_group = QGroupBox("System Console")
        console_layout = QVBoxLayout()
        self.txt_console = QTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setFont(QFont("Consolas", 9))
        self.txt_console.setStyleSheet("""
        QTextEdit {
            background-color: #000000;
            color: #00ff00;
            border: 1px solid #333333;
            font-family: 'Consolas';
        }
        """)
        self.txt_console.setMaximumHeight(150)
        console_layout.addWidget(self.txt_console)
        console_group.setLayout(console_layout)
        layout.addWidget(console_group)
        
        # Tab Widget for organized controls
        tab_widget = QTabWidget()
        
        # Tab 1: Training
        training_tab = self._create_training_tab()
        tab_widget.addTab(training_tab, "Training")
        
        # Tab 2: Generation
        generation_tab = self._create_generation_tab()
        tab_widget.addTab(generation_tab, "Generation")
        
        # Tab 3: Model Management
        model_tab = self._create_model_tab()
        tab_widget.addTab(model_tab, "Models")
        
        # Tab 4: System Monitoring
        monitor_tab = self._create_monitor_tab()
        tab_widget.addTab(monitor_tab, "Monitor")
        
        layout.addWidget(tab_widget)
        
        return panel
        
    def _create_training_tab(self) -> QWidget:
        """Create training configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # Corpus Selection
        corpus_group = QGroupBox("Corpus Selection")
        corpus_layout = QVBoxLayout()
        self.btn_load_corpus = QPushButton("Load Training Corpus")
        self.btn_load_corpus.setMinimumHeight(40)
        self.btn_load_corpus.clicked.connect(self.load_corpus)
        self.lbl_corpus = QLabel("No corpus loaded")
        self.lbl_corpus.setWordWrap(True)
        self.lbl_corpus.setStyleSheet("color: #888888; font-style: italic;")
        corpus_layout.addWidget(self.btn_load_corpus)
        corpus_layout.addWidget(self.lbl_corpus)
        corpus_group.setLayout(corpus_layout)
        layout.addWidget(corpus_group)
        
        # Training Parameters
        param_group = QGroupBox("Training Parameters")
        param_layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1000, 200000)
        self.spin_batch_size.setValue(20000)
        self.spin_batch_size.setSuffix(" tokens")
        batch_layout.addWidget(self.spin_batch_size)
        param_layout.addLayout(batch_layout)
        
        # Memory limit
        mem_layout = QHBoxLayout()
        mem_layout.addWidget(QLabel("Memory Limit:"))
        self.spin_memory_limit = QSpinBox()
        self.spin_memory_limit.setRange(100, 20000)
        self.spin_memory_limit.setValue(2000)
        self.spin_memory_limit.setSuffix(" MB")
        mem_layout.addWidget(self.spin_memory_limit)
        param_layout.addLayout(mem_layout)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Training Controls
        control_group = QGroupBox("Training Controls")
        control_layout = QVBoxLayout()
        self.btn_start_training = QPushButton("Start Training")
        self.btn_start_training.setMinimumHeight(45)
        self.btn_start_training.setStyleSheet("""
        QPushButton {
            background-color: #2e7d32;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #1b5e20;
        }
        """)
        self.btn_start_training.clicked.connect(self.start_training)
        self.btn_stop_training = QPushButton("Stop Training")
        self.btn_stop_training.setMinimumHeight(35)
        self.btn_stop_training.setEnabled(False)
        self.btn_stop_training.clicked.connect(self.stop_training)
        self.btn_reset_engine = QPushButton("Reset Engine")
        self.btn_reset_engine.setMinimumHeight(35)
        self.btn_reset_engine.setStyleSheet("background-color: #5d4037; color: white;")
        self.btn_reset_engine.clicked.connect(self.reset_engine)
        control_layout.addWidget(self.btn_start_training)
        control_layout.addWidget(self.btn_stop_training)
        control_layout.addWidget(self.btn_reset_engine)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Progress Display
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        self.lbl_training_status = QLabel("Idle")
        self.lbl_training_status.setStyleSheet("font-weight: bold;")
        self.progress_training = QProgressBar()
        self.progress_training.setRange(0, 100)
        self.progress_training.setTextVisible(True)
        self.lbl_training_stats = QLabel("")
        self.lbl_training_stats.setStyleSheet("color: #888888; font-size: 10pt;")
        progress_layout.addWidget(self.lbl_training_status)
        progress_layout.addWidget(self.progress_training)
        progress_layout.addWidget(self.lbl_training_stats)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Spacer
        layout.addStretch()
        
        return tab
        
    def _create_generation_tab(self) -> QWidget:
        """Create text generation tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # Seed Input
        seed_group = QGroupBox("Seed Text")
        seed_layout = QVBoxLayout()
        self.txt_seed = QTextEdit()
        self.txt_seed.setPlaceholderText("Enter seed text (e.g., 'To be or not to be')")
        self.txt_seed.setMaximumHeight(100)
        self.txt_seed.setFont(QFont("Consolas", 10))
        seed_layout.addWidget(self.txt_seed)
        seed_group.setLayout(seed_layout)
        layout.addWidget(seed_group)
        
        # Generation Parameters
        param_group = QGroupBox("Generation Parameters")
        param_layout = QVBoxLayout()
        
        # Temperature
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.slider_temp = QSlider(Qt.Orientation.Horizontal)
        self.slider_temp.setRange(10, 200)
        self.slider_temp.setValue(85)
        self.slider_temp.valueChanged.connect(self._update_temp_label)
        temp_layout.addWidget(self.slider_temp)
        self.lbl_temp_value = QLabel("0.85")
        self.lbl_temp_value.setMinimumWidth(40)
        temp_layout.addWidget(self.lbl_temp_value)
        param_layout.addLayout(temp_layout)
        
        # Top-P
        top_p_layout = QHBoxLayout()
        top_p_layout.addWidget(QLabel("Top-P:"))
        self.slider_top_p = QSlider(Qt.Orientation.Horizontal)
        self.slider_top_p.setRange(10, 100)
        self.slider_top_p.setValue(90)
        self.slider_top_p.valueChanged.connect(self._update_top_p_label)
        top_p_layout.addWidget(self.slider_top_p)
        self.lbl_top_p_value = QLabel("0.90")
        self.lbl_top_p_value.setMinimumWidth(40)
        top_p_layout.addWidget(self.lbl_top_p_value)
        param_layout.addLayout(top_p_layout)
        
        # Repetition Penalty
        rep_layout = QHBoxLayout()
        rep_layout.addWidget(QLabel("Repetition Penalty:"))
        self.slider_rep_penalty = QSlider(Qt.Orientation.Horizontal)
        self.slider_rep_penalty.setRange(10, 30)
        self.slider_rep_penalty.setValue(12)
        self.slider_rep_penalty.valueChanged.connect(self._update_rep_penalty_label)
        rep_layout.addWidget(self.slider_rep_penalty)
        self.lbl_rep_penalty_value = QLabel("1.2")
        self.lbl_rep_penalty_value.setMinimumWidth(40)
        rep_layout.addWidget(self.lbl_rep_penalty_value)
        param_layout.addLayout(rep_layout)
        
        # Length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Output Length:"))
        self.spin_output_length = QSpinBox()
        self.spin_output_length.setRange(50, 2000)
        self.spin_output_length.setValue(200)
        self.spin_output_length.setSuffix(" tokens")
        length_layout.addWidget(self.spin_output_length)
        param_layout.addLayout(length_layout)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Advanced Features
        advanced_group = QGroupBox("Advanced Features")
        advanced_layout = QVBoxLayout()
        self.cb_use_astar = QCheckBox("Enable A* Pathfinding")
        self.cb_use_momentum = QCheckBox("Enable Geometric Momentum")
        self.cb_use_tunnels = QCheckBox("Enable Trajectory Tunnels")
        self.cb_use_momentum.setChecked(True)
        self.cb_use_tunnels.setChecked(True)
        advanced_layout.addWidget(self.cb_use_astar)
        advanced_layout.addWidget(self.cb_use_momentum)
        advanced_layout.addWidget(self.cb_use_tunnels)
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # Generation Controls
        gen_control_group = QGroupBox("Generation Controls")
        gen_control_layout = QVBoxLayout()
        self.btn_generate = QPushButton("Generate Text")
        self.btn_generate.setMinimumHeight(50)
        self.btn_generate.setStyleSheet("""
        QPushButton {
            background-color: #1565c0;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            font-size: 12pt;
        }
        QPushButton:hover {
            background-color: #0d47a1;
        }
        """)
        self.btn_generate.clicked.connect(self.generate_text)
        self.btn_analyze = QPushButton("Analyze Text")
        self.btn_analyze.setMinimumHeight(35)
        self.btn_analyze.clicked.connect(self.analyze_text)
        self.btn_clear_output = QPushButton("Clear Output")
        self.btn_clear_output.setMinimumHeight(35)
        self.btn_clear_output.clicked.connect(self.clear_output)
        gen_control_layout.addWidget(self.btn_generate)
        gen_control_layout.addWidget(self.btn_analyze)
        gen_control_layout.addWidget(self.btn_clear_output)
        gen_control_group.setLayout(gen_control_layout)
        layout.addWidget(gen_control_group)
        
        # Generated Output Preview
        output_group = QGroupBox("Generated Output Preview")
        output_layout = QVBoxLayout()
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setFont(QFont("Consolas", 10))
        self.txt_output.setStyleSheet("""
        QTextEdit {
            background-color: #1a1a2e;
            color: #e0e0e0;
            border: 1px solid #333344;
        }
        """)
        self.txt_output.setMaximumHeight(150)
        output_layout.addWidget(self.txt_output)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        return tab
        
    def _create_model_tab(self) -> QWidget:
        """Create model management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # Model Browser
        browser_group = QGroupBox("Saved Models")
        browser_layout = QVBoxLayout()
        self.list_models = QListWidget()
        self.list_models.setFont(QFont("Consolas", 9))
        self.list_models.itemDoubleClicked.connect(self.load_selected_model)
        browser_layout.addWidget(self.list_models)
        browser_group.setLayout(browser_layout)
        layout.addWidget(browser_group)
        
        # Model Controls
        model_control_group = QGroupBox("Model Operations")
        model_control_layout = QVBoxLayout()
        self.btn_refresh_models = QPushButton("Refresh List")
        self.btn_refresh_models.clicked.connect(self.refresh_model_list)
        self.btn_load_model = QPushButton("Load Selected Model")
        self.btn_load_model.clicked.connect(self.load_selected_model)
        self.btn_delete_model = QPushButton("Delete Selected Model")
        self.btn_delete_model.setStyleSheet("background-color: #5d4037; color: white;")
        self.btn_delete_model.clicked.connect(self.delete_selected_model)
        self.btn_export_model = QPushButton("Export Model")
        self.btn_export_model.clicked.connect(self.export_model)
        model_control_layout.addWidget(self.btn_refresh_models)
        model_control_layout.addWidget(self.btn_load_model)
        model_control_layout.addWidget(self.btn_delete_model)
        model_control_layout.addWidget(self.btn_export_model)
        model_control_group.setLayout(model_control_layout)
        layout.addWidget(model_control_group)
        
        # Model Information
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout()
        self.txt_model_info = QTextEdit()
        self.txt_model_info.setReadOnly(True)
        self.txt_model_info.setMaximumHeight(150)
        self.txt_model_info.setFont(QFont("Consolas", 9))
        info_layout.addWidget(self.txt_model_info)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Refresh model list
        self.refresh_model_list()
        
        return tab
        
    def _create_monitor_tab(self) -> QWidget:
        """Create system monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # System Statistics
        stats_group = QGroupBox("Live System Statistics")
        stats_layout = QVBoxLayout()
        self.table_stats = QTableWidget(7, 2)
        self.table_stats.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table_stats.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_stats.verticalHeader().setVisible(False)
        self.table_stats.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Initialize rows
        stats_data = [
            ["Tokens Processed", "0"],
            ["N-Grams Learned", "0"],
            ["Memory Usage", "0 MB"],
            ["Voxel Density", "0%"],
            ["Training Time", "0s"],
            ["Last Optimization", "Never"],
            ["System Uptime", "00:00:00"]
        ]
        
        for i, (metric, value) in enumerate(stats_data):
            self.table_stats.setItem(i, 0, QTableWidgetItem(metric))
            self.table_stats.setItem(i, 1, QTableWidgetItem(value))
            
        stats_layout.addWidget(self.table_stats)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # N-Gram Distribution
        ngram_group = QGroupBox("N-Gram Distribution")
        ngram_layout = QVBoxLayout()
        self.txt_ngram_stats = QTextEdit()
        self.txt_ngram_stats.setReadOnly(True)
        self.txt_ngram_stats.setMaximumHeight(100)
        self.txt_ngram_stats.setFont(QFont("Consolas", 9))
        ngram_layout.addWidget(self.txt_ngram_stats)
        ngram_group.setLayout(ngram_layout)
        layout.addWidget(ngram_group)
        
        # System Controls
        system_group = QGroupBox("System Controls")
        system_layout = QVBoxLayout()
        self.btn_optimize_memory = QPushButton("Optimize Memory")
        self.btn_optimize_memory.clicked.connect(self.optimize_memory)
        self.btn_update_stats = QPushButton("Update Statistics")
        self.btn_update_stats.clicked.connect(self.update_statistics)
        self.btn_clear_stats = QPushButton("Clear Statistics")
        self.btn_clear_stats.clicked.connect(self.clear_statistics)
        system_layout.addWidget(self.btn_optimize_memory)
        system_layout.addWidget(self.btn_update_stats)
        system_layout.addWidget(self.btn_clear_stats)
        system_group.setLayout(system_layout)
        layout.addWidget(system_group)
        
        return tab
        
    def _create_visualization_panel(self) -> QWidget:
        """Create the visualization panel with 3D plot and output"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 3D Visualization Header
        viz_header = QLabel("Neural Voxel Cloud & Inference Trajectory")
        viz_header_font = QFont("Consolas", 12, QFont.Weight.Bold)
        viz_header.setFont(viz_header_font)
        viz_header.setStyleSheet("color: #00ffff; padding: 5px;")
        viz_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(viz_header)
        
        # Visualization Controls
        viz_controls = QHBoxLayout()
        
        # Density control
        viz_controls.addWidget(QLabel("Cloud Density:"))
        self.slider_density = QSlider(Qt.Orientation.Horizontal)
        self.slider_density.setRange(1, 100)
        self.slider_density.setValue(30)
        self.slider_density.valueChanged.connect(self._update_density_label)
        viz_controls.addWidget(self.slider_density)
        self.lbl_density_value = QLabel("30%")
        self.lbl_density_value.setMinimumWidth(40)
        viz_controls.addWidget(self.lbl_density_value)
        
        # Color map selection
        viz_controls.addWidget(QLabel("Color Map:"))
        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'cool'])
        self.combo_colormap.setCurrentText('viridis')
        self.combo_colormap.currentTextChanged.connect(self.refresh_visualization)
        viz_controls.addWidget(self.combo_colormap)
        
        # Refresh button
        self.btn_refresh_viz = QPushButton("Refresh")
        self.btn_refresh_viz.clicked.connect(self.refresh_visualization)
        viz_controls.addWidget(self.btn_refresh_viz)
        
        # Clear visualization button
        self.btn_clear_viz = QPushButton("Clear View")
        self.btn_clear_viz.clicked.connect(self.clear_visualization)
        viz_controls.addWidget(self.btn_clear_viz)
        
        viz_controls.addStretch()
        layout.addLayout(viz_controls)
        
        # 3D Plot Canvas
        self.fig = Figure(facecolor='#0a0a0a', figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #0a0a0a;")
        
        # Create 3D axis
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='#0a0a0a')
        
        # Configure axis appearance
        self.ax.grid(False)
        self.ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        self.ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        self.ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        
        # Set axis colors to white
        self.ax.xaxis.line.set_color("white")
        self.ax.yaxis.line.set_color("white")
        self.ax.zaxis.line.set_color("white")
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        
        # Set labels
        self.ax.set_xlabel('X (Char 1)', color='white')
        self.ax.set_ylabel('Y (Char 2)', color='white')
        self.ax.set_zlabel('Z (Char 3)', color='white')
        
        layout.addWidget(self.canvas)
        
        # Full Output Display
        output_group = QGroupBox("Complete Generated Output")
        output_layout = QVBoxLayout()
        self.txt_full_output = QTextEdit()
        self.txt_full_output.setReadOnly(False)  # Allow editing for copying
        self.txt_full_output.setFont(QFont("Georgia", 11))
        self.txt_full_output.setStyleSheet("""
        QTextEdit {
            background-color: #1e1e2e;
            color: #f0f0f0;
            border: 2px solid #333344;
            padding: 10px;
            line-height: 1.4;
        }
        """)
        output_layout.addWidget(self.txt_full_output)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Initial visualization
        self.refresh_visualization()
        
        return panel
        
    # ============================================================================
    # UI UPDATE METHODS
    # ============================================================================
    def _update_temp_label(self):
        """Update temperature label"""
        value = self.slider_temp.value() / 100.0
        self.lbl_temp_value.setText(f"{value:.2f}")
        
    def _update_top_p_label(self):
        """Update top-p label"""
        value = self.slider_top_p.value() / 100.0
        self.lbl_top_p_value.setText(f"{value:.2f}")
        
    def _update_rep_penalty_label(self):
        """Update repetition penalty label"""
        value = self.slider_rep_penalty.value() / 10.0
        self.lbl_rep_penalty_value.setText(f"{value:.1f}")
        
    def _update_density_label(self):
        """Update density label"""
        value = self.slider_density.value()
        self.lbl_density_value.setText(f"{value}%")
        
    def _system_log(self, message: str, level: str = "INFO"):
        """Log message to system console with timestamp and color coding"""
        timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
        # Color coding based on level
        if level == "ERROR":
            color = "#ff4444"
            prefix = "ERROR"
        elif level == "WARNING":
            color = "#ffaa00"
            prefix = "WARNING"
        elif level == "SUCCESS":
            color = "#44ff44"
            prefix = "SUCCESS"
        else:
            color = "#00aaff"
            prefix = "INFO"
            
        # Format message
        formatted = f'<span style="color: #888888;">{timestamp}</span> ' \
                   f'<span style="color: {color}; font-weight: bold;">{prefix}</span>: ' \
                   f'<span style="color: #dddddd;">{message}</span>'
                   
        # Append to console
        self.txt_console.append(formatted)
        
        # Auto-scroll to bottom
        scrollbar = self.txt_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Update status bar
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f"{prefix}: {message}", 5000)
            
    def _update_status_bar(self):
        """Update status bar with current statistics"""
        if hasattr(self, 'engine'):
            uptime = int(time.time() - self.engine.start_time)
            uptime_str = f"{uptime//3600:02d}:{(uptime%3600)//60:02d}:{uptime%60:02d}"
            
            stats = f"Tokens: {self.engine.tokens_processed:,} | " \
                   f"N-Grams: {sum(self.engine.ngrams_learned.values()):,} | " \
                   f"Memory: {self.engine.memory_usage_mb:.1f} MB | " \
                   f"Uptime: {uptime_str}"
            self.status_bar.showMessage(stats)
            
    def update_statistics(self):
        """Update statistics display"""
        if not hasattr(self, 'engine'):
            return
            
        # Update table
        self.table_stats.item(0, 1).setText(f"{self.engine.tokens_processed:,}")
        self.table_stats.item(1, 1).setText(f"{sum(self.engine.ngrams_learned.values()):,}")
        self.table_stats.item(2, 1).setText(f"{self.engine.memory_usage_mb:.1f} MB")
        
        # Calculate voxel density
        voxel_density = np.sum(self.engine.voxel_tensor > 0) / self.engine.voxel_tensor.size * 100
        self.table_stats.item(3, 1).setText(f"{voxel_density:.1f}%")
        
        # Training time
        training_time = int(self.engine.training_time)
        time_str = f"{training_time//3600:02d}:{(training_time%3600)//60:02d}:{training_time%60:02d}"
        self.table_stats.item(4, 1).setText(time_str)
        
        # Last optimization
        last_opt_str = self.engine.last_optimization.strftime("%H:%M:%S")
        self.table_stats.item(5, 1).setText(last_opt_str)
        
        # System uptime
        uptime = int(time.time() - self.engine.start_time)
        uptime_str = f"{uptime//3600:02d}:{(uptime%3600)//60:02d}:{uptime%60:02d}"
        self.table_stats.item(6, 1).setText(uptime_str)
        
        # N-Gram distribution
        ngram_text = ""
        for n in range(2, 6):
            count = self.engine.ngrams_learned.get(n, 0)
            ngram_text += f"{n}-grams: {count:,}\n"
        self.txt_ngram_stats.setText(ngram_text)
        
        self._system_log("Statistics updated")
        
    def clear_statistics(self):
        """Clear statistics display"""
        for i in range(self.table_stats.rowCount()):
            self.table_stats.item(i, 1).setText("0")
        self.txt_ngram_stats.clear()
        self._system_log("Statistics cleared")
        
    # ============================================================================
    # CORE FUNCTIONALITY
    # ============================================================================
    def load_corpus(self):
        """Load training corpus"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Corpus",
            "",
            "Text Files (*.txt);;All Files (*.*)"
        )
        if file_path:
            try:
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                if file_size > 50:
                    reply = QMessageBox.warning(
                        self,
                        "Large File Warning",
                        f"This file is {file_size:.1f} MB. Loading very large files may impact performance.\n"
                        "Proceed anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                        
                self.current_corpus = file_path
                self.lbl_corpus.setText(f"File: {os.path.basename(file_path)} ({file_size:.1f} MB)")
                self._system_log(f"Corpus loaded: {os.path.basename(file_path)}")
            except Exception as e:
                self._system_log(f"Error loading corpus: {str(e)}", "ERROR")
                
    def start_training(self):
        """Start training process"""
        if not hasattr(self, 'current_corpus') or not self.current_corpus:
            QMessageBox.warning(self, "No Corpus", "Please load a training corpus first.")
            return
            
        # Disable controls
        self.btn_start_training.setEnabled(False)
        self.btn_stop_training.setEnabled(True)
        self.btn_load_corpus.setEnabled(False)
        
        # Reset progress
        self.progress_training.setValue(0)
        self.lbl_training_status.setText("Training in progress...")
        
        # Create and start worker
        self.training_worker = TrainingWorker(self.engine, self.current_corpus)
        self.training_worker.progress_updated.connect(self.on_training_progress)
        self.training_worker.training_complete.connect(self.on_training_complete)
        self.training_worker.error_occurred.connect(self.on_training_error)
        self.training_worker.stats_updated.connect(self.on_training_stats_updated)
        self.training_worker.start()
        self._system_log("Training started", "INFO")
        
    def stop_training(self):
        """Stop training process"""
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            self.training_worker.wait()
            self.btn_start_training.setEnabled(True)
            self.btn_stop_training.setEnabled(False)
            self.btn_load_corpus.setEnabled(True)
            self.lbl_training_status.setText("Training stopped")
            self._system_log("Training stopped by user", "WARNING")
            
    def on_training_progress(self, progress: int, message: str):
        """Handle training progress updates"""
        self.progress_training.setValue(progress)
        self.lbl_training_status.setText(message)
        
    def on_training_stats_updated(self, stats: dict):
        """Update training statistics"""
        self.engine.tokens_processed = stats['tokens_processed']
        self.engine.ngrams_learned = stats['ngrams_learned']
        self.engine.memory_usage_mb = stats['memory_usage_mb']
        stats_text = f"Tokens: {stats['tokens_processed']:,} | " \
                    f"N-Grams: {sum(stats['ngrams_learned'].values()):,} | " \
                    f"Memory: {stats['memory_usage_mb']:.1f} MB"
        self.lbl_training_stats.setText(stats_text)
        
    def on_training_complete(self, message: str):
        """Handle training completion"""
        self.btn_start_training.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        self.btn_load_corpus.setEnabled(True)
        self.lbl_training_status.setText("Training complete")
        self.progress_training.setValue(100)
        
        # Update statistics
        self.update_statistics()
        
        # Refresh model list
        self.refresh_model_list()
        self._system_log(message, "SUCCESS")
        QMessageBox.information(self, "Training Complete", message)
        
    def on_training_error(self, error_message: str):
        """Handle training errors"""
        self.btn_start_training.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        self.btn_load_corpus.setEnabled(True)
        self.lbl_training_status.setText("Training failed")
        self.progress_training.setValue(0)
        self._system_log(error_message, "ERROR")
        QMessageBox.critical(self, "Training Error", error_message)
        
    def reset_engine(self):
        """Reset the engine to initial state"""
        reply = QMessageBox.question(
            self,
            "Reset Engine",
            "This will delete all learned patterns and reset the engine.\n"
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.engine.reset_engine()
            self.refresh_visualization()
            self.clear_output()
            self.clear_statistics()
            self._system_log("Engine reset complete", "SUCCESS")
            
    def generate_text(self):
        """Generate text based on current parameters"""
        seed_text = self.txt_seed.toPlainText().strip()
        if not seed_text:
            QMessageBox.warning(self, "No Seed", "Please enter some seed text.")
            return
            
        # Get parameters
        temperature = self.slider_temp.value() / 100.0
        top_p = self.slider_top_p.value() / 100.0
        repetition_penalty = self.slider_rep_penalty.value() / 10.0
        max_length = self.spin_output_length.value()
        use_astar = self.cb_use_astar.isChecked()
        use_momentum = self.cb_use_momentum.isChecked()
        
        # Disable button during generation
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("Generating...")
        
        # Create and start generation worker
        self.generation_worker = GenerationWorker(
            self.engine,
            seed_text,
            max_length,
            temperature,
            top_p,
            repetition_penalty,
            use_astar,
            use_momentum
        )
        self.generation_worker.generation_progress.connect(self.on_generation_progress)
        self.generation_worker.generation_complete.connect(self.on_generation_complete)
        self.generation_worker.start()
        self._system_log(f"Generating text with seed: '{seed_text[:50]}...'")
        
    def on_generation_progress(self, progress: int, message: str):
        """Handle generation progress updates"""
        self.btn_generate.setText(f"Generating... {progress}%")
        
    def on_generation_complete(self, generated_text: str, trajectory: list, metrics: dict):
        """Handle generation completion"""
        # Re-enable button
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("Generate Text")
        
        if generated_text.startswith("Error:"):
            self._system_log(generated_text, "ERROR")
            QMessageBox.critical(self, "Generation Error", generated_text)
            return
            
        # Update output displays
        self.txt_output.setText(generated_text[:500] + ("..." if len(generated_text) > 500 else ""))
        self.txt_full_output.setText(generated_text)
        self.last_generated_text = generated_text
        
        # Update trajectory and visualization
        self.current_trajectory = trajectory
        self.refresh_visualization()
        
        # Log success
        self._system_log(f"Generated {len(generated_text):,} characters", "SUCCESS")
        
        # Show brief metrics in status bar
        if metrics:
            rep_rate = metrics.get('repetition_rate', 0) * 100
            self.status_bar.showMessage(
                f"Generation complete | Repetition: {rep_rate:.1f}% | "
                f"Avg Word Length: {metrics.get('avg_word_length', 0):.1f}"
            )
            
    def analyze_text(self):
        """Analyze the generated text"""
        if not hasattr(self, 'last_generated_text') or not self.last_generated_text:
            QMessageBox.warning(self, "No Text", "Please generate text first.")
            return
            
        # Get metrics from engine
        tokens = self.engine.tokenizer.tokenize(self.last_generated_text)
        metrics = self.engine.tokenizer.analyze_text_metrics(tokens)
        
        # Convert to dict for dialog
        metrics_dict = {
            'perplexity': metrics.perplexity,
            'coherence_score': metrics.coherence_score,
            'diversity_score': metrics.diversity_score,
            'repetition_rate': metrics.repetition_rate,
            'avg_word_length': metrics.avg_word_length,
            'sentence_complexity': metrics.sentence_complexity
        }
        
        # Show analysis dialog
        dialog = TextAnalysisDialog(metrics_dict, self)
        dialog.exec()
        
    def clear_output(self):
        """Clear all output displays"""
        self.txt_output.clear()
        self.txt_full_output.clear()
        self.current_trajectory = []
        self.refresh_visualization()
        self._system_log("Output cleared")
        
    def refresh_visualization(self):
        """Refresh the 3D visualization"""
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Set axis properties again (clearing removes them)
            self.ax.set_facecolor('#0a0a0a')
            self.ax.grid(False)
            self.ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            self.ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            self.ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            self.ax.xaxis.line.set_color("white")
            self.ax.yaxis.line.set_color("white")
            self.ax.zaxis.line.set_color("white")
            self.ax.tick_params(colors='white')
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.zaxis.label.set_color('white')
            self.ax.set_xlabel('X (Char 1)', color='white')
            self.ax.set_ylabel('Y (Char 2)', color='white')
            self.ax.set_zlabel('Z (Char 3)', color='white')
            
            # Get density threshold from slider
            density_percent = self.slider_density.value()
            max_density = np.max(self.engine.voxel_tensor) if np.any(self.engine.voxel_tensor) else 1.0
            threshold = max_density * (1.0 - density_percent / 100.0)
            
            # Create voxel cloud
            indices = np.where(self.engine.voxel_tensor > threshold)
            if len(indices[0]) > 0:
                # Limit points for performance
                max_points = 5000
                if len(indices[0]) > max_points:
                    # Random sampling
                    sample_idx = np.random.choice(
                        len(indices[0]),
                        min(max_points, len(indices[0])),
                        replace=False
                    )
                    x = indices[0][sample_idx]
                    y = indices[1][sample_idx]
                    z = indices[2][sample_idx]
                    c = self.engine.voxel_tensor[indices][0][sample_idx]
                else:
                    x, y, z = indices
                    if len(self.engine.voxel_tensor.shape) == 3:
                        c = self.engine.voxel_tensor[x, y, z]
                    else:
                        c = self.engine.voxel_tensor[indices]
                        
                # Get selected colormap
                cmap_name = self.combo_colormap.currentText()
                
                # Create scatter plot with color based on density
                scatter = self.ax.scatter(
                    x, y, z,
                    c=c,
                    cmap=cmap_name,
                    alpha=0.3,
                    s=1,
                    label="Voxel Cloud"
                )
                
            # Plot current trajectory if available
            if self.current_trajectory and len(self.current_trajectory) > 1:
                trajectory = np.array(self.current_trajectory)
                # Plot trajectory line
                self.ax.plot(
                    trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    color='#00ffff',
                    linewidth=2.5,
                    alpha=0.8,
                    label="Inference Trajectory"
                )
                
                # Plot start and end points
                self.ax.scatter(
                    trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                    color='#00ff00',
                    s=100,
                    label="Start",
                    edgecolors='white',
                    linewidth=1
                )
                self.ax.scatter(
                    trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                    color='#ff0000',
                    s=100,
                    label="End",
                    edgecolors='white',
                    linewidth=1
                )
                
            # Set viewing angle
            self.ax.view_init(elev=20, azim=45)
            
            # Add legend only if there are artists with labels
            if self.ax.get_legend_handles_labels()[0]:
                self.ax.legend(loc='upper right', fontsize='small', facecolor='#1a1a1a')
                
            # Redraw canvas
            self.canvas.draw()
            
        except Exception as e:
            self._system_log(f"Visualization error: {str(e)}", "ERROR")
            
    def clear_visualization(self):
        """Clear the visualization"""
        self.current_trajectory = []
        self.refresh_visualization()
        self._system_log("Visualization cleared")
        
    def refresh_model_list(self):
        """Refresh the list of saved models"""
        self.list_models.clear()
        self.loaded_models = []
        if not os.path.exists(self.engine.model_dir):
            os.makedirs(self.engine.model_dir, exist_ok=True)
            
        # Scan for model directories
        for item in os.listdir(self.engine.model_dir):
            item_path = os.path.join(self.engine.model_dir, item)
            if os.path.isdir(item_path):
                # Check if it's a valid model
                metadata_path = os.path.join(item_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        # Create list item
                        model_item = QListWidgetItem()
                        model_name = item
                        token_count = metadata.get('tokens_processed', 0)
                        save_time = metadata.get('save_time', 'Unknown')
                        model_text = f"Model: {model_name}\n"
                        model_text += f"   Tokens: {token_count:,} | "
                        model_text += f"Saved: {save_time[:19]}"
                        model_item.setText(model_text)
                        model_item.setData(Qt.ItemDataRole.UserRole, item_path)
                        self.list_models.addItem(model_item)
                        self.loaded_models.append(item_path)
                    except Exception as e:
                        continue
                        
        self._system_log(f"Found {self.list_models.count()} saved models")
        
    def load_selected_model(self):
        """Load the selected model"""
        selected_items = self.list_models.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a model to load.")
            return
            
        model_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", "Model directory not found.")
            return
            
        try:
            # Load model
            success = self.engine.load_model(model_path)
            if success:
                # Update UI
                self.refresh_visualization()
                self.update_statistics()
                # Update model info
                metadata_path = os.path.join(model_path, "metadata.json")
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                info_text = f"Model: {os.path.basename(model_path)}\n"
                info_text += f"Tokens: {metadata['tokens_processed']:,}\n"
                info_text += f"N-Grams: {sum(metadata['ngrams_learned'].values()):,}\n"
                info_text += f"Memory: {metadata.get('memory_usage_mb', 0):.1f} MB\n"
                info_text += f"Training Time: {metadata.get('training_time', 0):.1f} seconds\n"
                info_text += f"Saved: {metadata['save_time']}"
                self.txt_model_info.setText(info_text)
                self._system_log(f"Model loaded: {os.path.basename(model_path)}", "SUCCESS")
            else:
                self._system_log("Failed to load model", "ERROR")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{str(e)}")
            self._system_log(f"Load error: {str(e)}", "ERROR")
            
    def delete_selected_model(self):
        """Delete the selected model"""
        selected_items = self.list_models.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a model to delete.")
            return
            
        model_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self,
            "Delete Model",
            f"Are you sure you want to delete this model?\n{os.path.basename(model_path)}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Remove directory and contents
                import shutil
                shutil.rmtree(model_path)
                # Refresh list
                self.refresh_model_list()
                self._system_log(f"Model deleted: {os.path.basename(model_path)}", "SUCCESS")
            except Exception as e:
                QMessageBox.critical(self, "Delete Error", f"Failed to delete model:\n{str(e)}")
                self._system_log(f"Delete error: {str(e)}", "ERROR")
                
    def export_model(self):
        """Export current model"""
        save_path = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            ""
        )
        if save_path:
            try:
                export_path = self.engine.save_model(
                    f"export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                # Copy to selected directory
                import shutil
                export_name = os.path.basename(export_path)
                destination = os.path.join(save_path, export_name)
                shutil.copytree(export_path, destination)
                self._system_log(f"Model exported to: {destination}", "SUCCESS")
                QMessageBox.information(self, "Export Complete", f"Model exported to:\n{destination}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export model:\n{str(e)}")
                self._system_log(f"Export error: {str(e)}", "ERROR")
                
    def optimize_memory(self):
        """Run memory optimization"""
        try:
            self._system_log("Starting memory optimization...")
            # Run optimization
            self.engine._optimize_memory()
            # Update visualization and statistics
            self.refresh_visualization()
            self.update_statistics()
            self._system_log("Memory optimization complete", "SUCCESS")
        except Exception as e:
            self._system_log(f"Optimization error: {str(e)}", "ERROR")
            
    def save_current_model(self):
        """Save the current model"""
        try:
            model_name, ok = QInputDialog.getText(
                self,
                "Save Model",
                "Enter model name:",
                text=f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            if ok and model_name:
                save_path = self.engine.save_model(model_name)
                self.refresh_model_list()
                QMessageBox.information(self, "Save Complete", f"Model saved to:\n{save_path}")
                self._system_log(f"Model saved: {model_name}", "SUCCESS")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save model:\n{str(e)}")
            self._system_log(f"Save error: {str(e)}", "ERROR")
            
    def load_model_dialog(self):
        """Open dialog to load a model"""
        model_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Model Directory",
            self.engine.model_dir
        )
        if model_dir and os.path.exists(model_dir):
            try:
                # Check if it's a valid model
                metadata_path = os.path.join(model_dir, "metadata.json")
                if not os.path.exists(metadata_path):
                    QMessageBox.warning(self, "Invalid Model", 
                                      "Selected directory does not contain a valid model.")
                    return
                    
                # Load model
                success = self.engine.load_model(model_dir)
                if success:
                    self.refresh_visualization()
                    self.update_statistics()
                    self.refresh_model_list()
                    
                    # Update model info
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    info_text = f"Model: {os.path.basename(model_dir)}\n"
                    info_text += f"Tokens: {metadata['tokens_processed']:,}\n"
                    info_text += f"N-Grams: {sum(metadata['ngrams_learned'].values()):,}\n"
                    info_text += f"Memory: {metadata.get('memory_usage_mb', 0):.1f} MB\n"
                    info_text += f"Training Time: {metadata.get('training_time', 0):.1f} seconds\n"
                    info_text += f"Saved: {metadata['save_time']}"
                    self.txt_model_info.setText(info_text)
                    
                    self._system_log(f"Model loaded: {os.path.basename(model_dir)}", "SUCCESS")
                    QMessageBox.information(self, "Load Complete", 
                                          f"Model loaded successfully:\n{os.path.basename(model_dir)}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{str(e)}")
                self._system_log(f"Load error: {str(e)}", "ERROR")
                
    def export_generated_text(self):
        """Export the currently generated text"""
        if not hasattr(self, 'last_generated_text') or not self.last_generated_text:
            QMessageBox.warning(self, "No Text", "No generated text to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Generated Text",
            f"generated_text_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.last_generated_text)
                self._system_log(f"Text exported to: {file_path}", "SUCCESS")
                QMessageBox.information(self, "Export Complete", 
                                      f"Generated text saved to:\n{file_path}")
            except Exception as e:
                self._system_log(f"Export error: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Export Error", f"Failed to export text:\n{str(e)}")
                
    def compare_models_dialog(self):
        """Open dialog to compare multiple models"""
        # Get selected models
        selected_items = self.list_models.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Selection Error", 
                              "Please select at least 2 models to compare.")
            return
            
        model_paths = []
        for item in selected_items:
            model_path = item.data(Qt.ItemDataRole.UserRole)
            model_paths.append(model_path)
            
        try:
            comparison = self.engine.compare_models(model_paths)
            
            # Create comparison dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Model Comparison")
            dialog.setMinimumSize(800, 600)
            
            layout = QVBoxLayout()
            
            # Create table
            table = QTableWidget(len(comparison), 6)
            table.setHorizontalHeaderLabels([
                "Model Name", "Tokens Processed", "2-grams", "3-grams", 
                "Memory (MB)", "Training Time (s)"
            ])
            
            row = 0
            for model_name, stats in comparison.items():
                table.setItem(row, 0, QTableWidgetItem(model_name))
                
                if "error" in stats:
                    table.setItem(row, 1, QTableWidgetItem(f"Error: {stats['error']}"))
                    row += 1
                    continue
                    
                table.setItem(row, 1, QTableWidgetItem(f"{stats['tokens_processed']:,}"))
                
                # N-gram counts
                ngrams = stats.get("ngrams_learned", {})
                table.setItem(row, 2, QTableWidgetItem(f"{ngrams.get(2, 0):,}"))
                table.setItem(row, 3, QTableWidgetItem(f"{ngrams.get(3, 0):,}"))
                
                table.setItem(row, 4, QTableWidgetItem(f"{stats.get('memory_usage', 0):.1f}"))
                table.setItem(row, 5, QTableWidgetItem(f"{stats.get('training_time', 0):.1f}"))
                
                row += 1
                
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            layout.addWidget(table)
            
            # Add export button
            btn_export = QPushButton("Export Comparison")
            btn_export.clicked.connect(lambda: self.export_comparison(comparison))
            layout.addWidget(btn_export)
            
            # Add close button
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(dialog.accept)
            layout.addWidget(btn_close)
            
            dialog.setLayout(layout)
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Comparison Error", f"Failed to compare models:\n{str(e)}")
            self._system_log(f"Comparison error: {str(e)}", "ERROR")
            
    def export_comparison(self, comparison_data):
        """Export model comparison data"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Comparison Report",
            f"model_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    import csv
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Model Name", "Tokens Processed", "2-grams", "3-grams", 
                                       "4-grams", "5-grams", "Memory (MB)", "Training Time (s)"])
                        
                        for model_name, stats in comparison_data.items():
                            if "error" in stats:
                                writer.writerow([model_name, f"Error: {stats['error']}", "", "", "", "", "", ""])
                                continue
                                
                            ngrams = stats.get("ngrams_learned", {})
                            writer.writerow([
                                model_name,
                                stats.get('tokens_processed', 0),
                                ngrams.get(2, 0),
                                ngrams.get(3, 0),
                                ngrams.get(4, 0),
                                ngrams.get(5, 0),
                                stats.get('memory_usage', 0),
                                stats.get('training_time', 0)
                            ])
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("Model Comparison Report\n")
                        f.write("=" * 50 + "\n\n")
                        for model_name, stats in comparison_data.items():
                            f.write(f"Model: {model_name}\n")
                            if "error" in stats:
                                f.write(f"  Error: {stats['error']}\n")
                            else:
                                f.write(f"  Tokens Processed: {stats.get('tokens_processed', 0):,}\n")
                                f.write(f"  Memory Usage: {stats.get('memory_usage', 0):.1f} MB\n")
                                f.write(f"  Training Time: {stats.get('training_time', 0):.1f} seconds\n")
                                
                                ngrams = stats.get("ngrams_learned", {})
                                f.write("  N-Gram Distribution:\n")
                                for n in range(2, 6):
                                    count = ngrams.get(n, 0)
                                    f.write(f"    {n}-grams: {count:,}\n")
                            f.write("\n" + "-" * 50 + "\n\n")
                            
                self._system_log(f"Comparison exported to: {file_path}", "SUCCESS")
                QMessageBox.information(self, "Export Complete", 
                                      f"Comparison report saved to:\n{file_path}")
            except Exception as e:
                self._system_log(f"Export error: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Export Error", f"Failed to export comparison:\n{str(e)}")
                
    def open_batch_generation(self):
        """Open batch generation dialog"""
        dialog = BatchGenerationDialog(self.engine, self)
        if dialog.exec():
            if dialog.generated_files:
                reply = QMessageBox.question(
                    self,
                    "Batch Generation Complete",
                    f"Generated {len(dialog.generated_files)} texts.\n"
                    f"Would you like to open the output directory?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    output_dir = os.path.dirname(dialog.generated_files[0])
                    QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))
                    
    def open_documentation(self):
        """Open documentation"""
        doc_text = """
VoxelPath Production System v3.0 Documentation
================================================

Core Concepts:
- Hierarchical N-Gram Memory: The system learns patterns from 2-grams to 5-grams
- 3D Voxel Representation: Each token is mapped to a 3D coordinate space
- Geometric Momentum: Tracks movement patterns through the voxel space
- Trajectory Tunnels: Fast paths for frequently occurring patterns

Training:
1. Load a text corpus (recommended: 1-10 MB for optimal performance)
2. Configure batch size and memory limits
3. Start training - system will learn patterns and build the voxel space
4. Save the trained model for future use

Text Generation:
1. Enter seed text to start generation
2. Adjust parameters:
   - Temperature: Higher values increase creativity (0.1-2.0)
   - Top-P: Nucleus sampling threshold (0.1-1.0)
   - Repetition Penalty: Reduces repetitive text (1.0-3.0)
3. Enable advanced features as needed
4. Generate and analyze the output

Advanced Features:
- A* Pathfinding: Guides generation through high-density regions
- Geometric Momentum: Maintains coherent trajectory through space
- Trajectory Tunnels: Accelerates common patterns
- Batch Generation: Create multiple texts with varying parameters

Analysis Tools:
- Text Quality Metrics: Perplexity, coherence, diversity scores
- Model Comparison: Compare performance across different models
- Memory Optimization: Clean up and optimize model size

Best Practices:
- Start with smaller corpora for testing
- Use temperature around 0.85 for balanced creativity/coherence
- Apply repetition penalty between 1.2-1.5 for natural text
- Periodically optimize memory to maintain performance

For more information, visit the project repository or contact support.
        """
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Documentation")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Consolas", 10))
        text_edit.setText(doc_text)
        layout.addWidget(text_edit)
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)
        
        dialog.setLayout(layout)
        dialog.exec()
        
    def show_about_dialog(self):
        """Show about dialog"""
        about_text = """
VoxelPath Production System v3.0
=================================

A Geometric 3D Coordinate-Mapped Trajectory Model with 
Hierarchical N-Gram Memory and Stochastic Graph-Based Architecture

Developed with:
- Python 3.10+
- NumPy, Matplotlib
- PyQt6 for GUI
- qdarktheme for styling

Core Features:
- Hierarchical N-Gram learning (2-5 grams)
- 3D voxel space representation
- Advanced text generation with multiple sampling strategies
- Visual trajectory mapping
- Comprehensive analysis tools
- Batch processing capabilities

Copyright 2026 VoxelPath Systems
All rights reserved.

For support and documentation, please refer to the help section.
        """
        
        QMessageBox.about(self, "About VoxelPath", about_text)
        
    def closeEvent(self, event):
        """Handle application close event"""
        # Stop any running workers
        if self.training_worker and self.training_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Training in Progress",
                "Training is still in progress. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.training_worker.stop()
                self.training_worker.wait()
                event.accept()
            else:
                event.ignore()
                return
                
        # Save any unsaved model
        reply = QMessageBox.question(
            self,
            "Save Model",
            "Would you like to save the current model before exiting?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.engine.save_model()
                self._system_log("Model saved before exit", "SUCCESS")
            except Exception as e:
                self._system_log(f"Failed to save model: {str(e)}", "ERROR")
        if reply != QMessageBox.StandardButton.Cancel:
            event.accept()
        else:
            event.ignore()
            
# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    # Set application style
    app.setStyle('Fusion')
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    
    # Set application icon and metadata
    app.setApplicationName("VoxelPath Production System")
    app.setApplicationVersion("3.0.0")
    
    # Create and show main window
    window = VoxelPathProductionUI()
    window.show()
    
    # Start application
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()