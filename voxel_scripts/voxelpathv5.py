"""
VoxelPath Production System v3.2
A Geometric 3D Coordinate-Mapped Trajectory Model with Hierarchical N-Gram Memory,
Stochastic Graph-Based Architecture, Advanced Training Metrics, and Real-time Monitoring
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
import traceback
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
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
    QMenu, QMenuBar, QInputDialog, QDialog, QDialogButtonBox, QGridLayout,
    QSizePolicy, QScrollArea
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize, QUrl, QAbstractTableModel, QEvent
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QAction, QDesktopServices, QValidator, QPixmap, QPainter, QBrush
import qdarktheme
# Visualization Imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation

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
    last_updated: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def add_transition(self, word: str, weight: float = 1.0):
        """Add a weighted transition from this context"""
        self.next_words[word] = self.next_words.get(word, 0.0) + weight
        self.total_count += weight
        self.last_updated = datetime.datetime.now()
        
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
        
    def calculate_perplexity(self, validation_sequence: List[str]) -> float:
        """Calculate perplexity for a validation sequence"""
        if not validation_sequence or not self.next_words:
            return float('inf')
            
        log_prob_sum = 0.0
        count = 0
        
        for word in validation_sequence:
            prob = self.get_probability(word)
            if prob > 0:
                log_prob_sum += math.log(prob)
                count += 1
        
        if count == 0:
            return float('inf')
            
        avg_log_prob = log_prob_sum / count
        perplexity = math.exp(-avg_log_prob)
        return perplexity
        
@dataclass
class TrajectoryTunnel:
    """Fast path for frequently occurring patterns"""
    pattern: str
    continuation: str
    usage_count: int = 0
    last_used: Optional[datetime.datetime] = None
    confidence: float = 0.0
    creation_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def should_activate(self, min_confidence: float = 0.8) -> bool:
        """Check if tunnel should be used based on confidence"""
        return self.confidence >= min_confidence
        
    def update_confidence(self, was_correct: bool, learning_rate: float = 0.1):
        """Update tunnel confidence based on prediction accuracy"""
        if was_correct:
            self.confidence = min(1.0, self.confidence + learning_rate * (1.0 - self.confidence))
        else:
            self.confidence = max(0.0, self.confidence - learning_rate * self.confidence)
        self.last_used = datetime.datetime.now()
        self.usage_count += 1
        
@dataclass
class GeometricMomentum:
    """Track movement through voxel space"""
    position: Tuple[int, int, int]
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_update: datetime.datetime = field(default_factory=datetime.datetime.now)
    
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
        
        # Calculate time delta for acceleration
        time_delta = (datetime.datetime.now() - self.last_update).total_seconds()
        time_delta = max(time_delta, 0.001)  # Avoid division by zero
        
        # Update acceleration (second derivative)
        self.acceleration = (
            self.velocity[0] / time_delta if time_delta > 0 else 0.0,
            self.velocity[1] / time_delta if time_delta > 0 else 0.0,
            self.velocity[2] / time_delta if time_delta > 0 else 0.0
        )
        
        self.position = new_position
        self.last_update = datetime.datetime.now()
        
@dataclass
class TrainingMetrics:
    """Comprehensive training metrics and statistics"""
    epoch: int = 0
    tokens_processed: int = 0
    ngrams_learned: Dict[int, int] = field(default_factory=lambda: {n: 0 for n in range(2, 6)})
    perplexity: float = float('inf')
    coherence_score: float = 0.0
    diversity_score: float = 0.0
    memory_usage_mb: float = 0.0
    training_rate: float = 0.0  # tokens per second
    validation_loss: float = float('inf')
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'epoch': self.epoch,
            'tokens_processed': self.tokens_processed,
            'ngrams_learned': self.ngrams_learned.copy(),
            'perplexity': self.perplexity,
            'coherence_score': self.coherence_score,
            'diversity_score': self.diversity_score,
            'memory_usage_mb': self.memory_usage_mb,
            'training_rate': self.training_rate,
            'validation_loss': self.validation_loss,
            'timestamp': self.timestamp.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingMetrics':
        """Create from dictionary"""
        metrics = cls()
        metrics.epoch = data.get('epoch', 0)
        metrics.tokens_processed = data.get('tokens_processed', 0)
        metrics.ngrams_learned = data.get('ngrams_learned', {n: 0 for n in range(2, 6)})
        metrics.perplexity = data.get('perplexity', float('inf'))
        metrics.coherence_score = data.get('coherence_score', 0.0)
        metrics.diversity_score = data.get('diversity_score', 0.0)
        metrics.memory_usage_mb = data.get('memory_usage_mb', 0.0)
        metrics.training_rate = data.get('training_rate', 0.0)
        metrics.validation_loss = data.get('validation_loss', float('inf'))
        timestamp_str = data.get('timestamp', datetime.datetime.now().isoformat())
        metrics.timestamp = datetime.datetime.fromisoformat(timestamp_str)
        return metrics
        
@dataclass
class TextAnalysisResult:
    """Detailed text analysis metrics"""
    perplexity: float = float('inf')
    coherence_score: float = 0.0
    diversity_score: float = 0.0
    repetition_rate: float = 0.0
    avg_word_length: float = 0.0
    sentence_complexity: float = 0.0
    sentiment: float = 0.0  # -1.0 to 1.0
    topic_consistency: float = 0.0
    grammatical_correctness: float = 0.0
    
# ============================================================================
# TOKENIZATION ENGINE
# ============================================================================
class IntelligentTokenizer:
    """Advanced tokenizer with punctuation handling and ASCII sanitization"""
    # Regex patterns for sophisticated tokenization
    PUNCTUATION_PATTERN = r'([\.,!?;:"\'\-–—()\[\]{}])'
    WORD_PATTERN = r'(\b[A-Za-z]+\b)'
    NUMBER_PATTERN = r'(\b\d+\b)'
    
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
        '»': '"',
        'é': 'e',
        'è': 'e',
        'ê': 'e',
        'ë': 'e',
        'á': 'a',
        'à': 'a',
        'â': 'a',
        'ä': 'a',
        'í': 'i',
        'ì': 'i',
        'î': 'i',
        'ï': 'i',
        'ó': 'o',
        'ò': 'o',
        'ô': 'o',
        'ö': 'o',
        'ú': 'u',
        'ù': 'u',
        'û': 'u',
        'ü': 'u',
        'ñ': 'n',
        'ç': 'c'
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
        text = ''.join(char for char in text if 31 < ord(char) < 127)
        return text
        
    def tokenize(self, text: str) -> List[str]:
        """Intelligent tokenization preserving punctuation as separate tokens"""
        text = self.sanitize_text(text)
        tokens = []
        # Use regex to split while keeping delimiters
        matches = self.token_pattern.finditer(text)
        for match in matches:
            token = match.group()
            if token and token.strip():  # Filter out empty matches and whitespace
                tokens.append(token.lower())  # Case normalization
                
        # Post-processing: handle contractions
        processed_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] in ["'", "’"] and tokens[i+1] in ['t', 's', 'm', 'd', 'll', 've', 're']:
                # Merge contractions like "don't", "it's", etc.
                if processed_tokens:
                    processed_tokens[-1] = processed_tokens[-1] + "'" + tokens[i+1]
                i += 2
            else:
                processed_tokens.append(tokens[i])
                i += 1
                
        return processed_tokens
        
    def detokenize(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens with proper spacing and capitalization"""
        result = []
        capitalize_next = True
        
        for i, token in enumerate(tokens):
            # Handle sentence boundaries
            if token in ['.', '!', '?'] and i < len(tokens) - 1:
                result.append(token)
                capitalize_next = True
            elif capitalize_next and token.isalpha():
                result.append(token.capitalize())
                capitalize_next = False
            elif token in self._punctuation_no_space_before:
                # Attach punctuation without space
                if result:
                    result[-1] = result[-1] + token
            elif result and result[-1] and result[-1][-1] in self._punctuation_no_space_after:
                # Attach to opening punctuation without space
                result[-1] = result[-1] + token
            else:
                # Add space before word/number
                if result:
                    result.append(' ' + token)
                else:
                    result.append(token)
                    capitalize_next = False
            
            # Handle special cases for apostrophes
            if token == "'" and i > 0 and i < len(tokens) - 1:
                if result:
                    result[-1] = result[-1] + "'"
                    continue
            
        # Capitalize first letter if needed
        if result and result[0]:
            result[0] = result[0].capitalize()
            
        return ''.join(result).strip()
        
    def analyze_text(self, tokens: List[str]) -> TextAnalysisResult:
        """Comprehensive text analysis with multiple quality metrics"""
        result = TextAnalysisResult()
        
        # Calculate average word length (excluding punctuation)
        words = [token for token in tokens if token.isalpha()]
        if words:
            word_lengths = [len(word) for word in words]
            result.avg_word_length = sum(word_lengths) / len(word_lengths)
            
        # Calculate repetition rate
        if tokens:
            unique_tokens = set(tokens)
            result.repetition_rate = 1.0 - (len(unique_tokens) / len(tokens))
            
        # Calculate sentence complexity
        sentence_delimiters = ['.', '!', '?']
        sentences = []
        current_sentence = []
        
        for token in tokens:
            current_sentence.append(token)
            if token in sentence_delimiters:
                sentences.append(current_sentence)
                current_sentence = []
                
        if current_sentence:  # Add last incomplete sentence
            sentences.append(current_sentence)
            
        if sentences:
            sentence_lengths = [len(s) for s in sentences]
            result.sentence_complexity = sum(sentence_lengths) / len(sentence_lengths)
            
        # Calculate vocabulary diversity (type-token ratio)
        if tokens:
            # Moving window diversity
            window_size = min(50, len(tokens))
            diversity_scores = []
            
            for i in range(0, len(tokens), window_size // 2):
                window = tokens[i:i+window_size]
                if window:
                    unique_in_window = len(set(window))
                    diversity_scores.append(unique_in_window / len(window))
                    
            if diversity_scores:
                result.diversity_score = sum(diversity_scores) / len(diversity_scores)
                
        # Calculate topic consistency (simplified version)
        # Group related words by common prefixes or semantic similarity
        if words:
            # Simple prefix-based grouping
            prefix_groups = defaultdict(list)
            for word in words:
                if len(word) >= 3:
                    prefix = word[:3]
                    prefix_groups[prefix].append(word)
                    
            # Calculate consistency score based on dominant groups
            if prefix_groups:
                group_sizes = [len(group) for group in prefix_groups.values()]
                max_group = max(group_sizes) if group_sizes else 0
                total_words = len(words)
                if total_words > 0:
                    result.topic_consistency = max_group / total_words
                
        # Grammatical correctness estimation (very simplified)
        # Check for basic patterns like subject-verb agreement
        result.grammatical_correctness = self._estimate_grammatical_correctness(tokens)
        
        # Coherence score (transitions between sentences)
        result.coherence_score = self._calculate_coherence_score(sentences)
        
        # Perplexity estimation
        result.perplexity = self._estimate_perplexity(tokens)
        
        return result
        
    def _estimate_grammatical_correctness(self, tokens: List[str]) -> float:
        """Simple grammatical correctness estimation"""
        score = 1.0
        
        # Check for proper sentence structure
        sentence_starts = []
        sentence_ends = []
        
        for i, token in enumerate(tokens):
            if i == 0 or tokens[i-1] in ['.', '!', '?']:
                sentence_starts.append(token)
            if token in ['.', '!', '?']:
                sentence_ends.append(token)
                
        # Check ratio of sentence starts to ends
        if sentence_starts and sentence_ends:
            ratio = len(sentence_ends) / len(sentence_starts)
            score *= max(0.5, min(1.5, ratio))  # Normalize to 0.5-1.5 range
            
        # Check for common grammatical errors
        error_patterns = [
            (r'\bthe the\b', 0.1),
            (r'\ba a\b', 0.1),
            (r'\ban an\b', 0.1),
            (r'\bis is\b', 0.2),
            (r'\bare are\b', 0.2),
            (r'\bwas was\b', 0.2),
            (r'\bwere were\b', 0.2),
        ]
        
        text = self.detokenize(tokens).lower()
        for pattern, penalty in error_patterns:
            if re.search(pattern, text):
                score *= (1 - penalty)
                
        return max(0.0, min(1.0, score))
        
    def _calculate_coherence_score(self, sentences: List[List[str]]) -> float:
        """Calculate coherence between sentences"""
        if len(sentences) < 2:
            return 1.0
            
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            current_sentence = sentences[i]
            next_sentence = sentences[i+1]
            
            # Extract nouns and verbs from each sentence
            current_words = [w for w in current_sentence if w.isalpha() and len(w) > 2]
            next_words = [w for w in next_sentence if w.isalpha() and len(w) > 2]
            
            # Simple overlap scoring
            common_words = set(current_words) & set(next_words)
            if current_words and next_words:
                overlap = len(common_words) / (len(current_words) + len(next_words))
                coherence_scores.append(overlap)
                
        if coherence_scores:
            return min(1.0, sum(coherence_scores) / len(coherence_scores) * 2)  # Scale up
        return 0.5
        
    def _estimate_perplexity(self, tokens: List[str]) -> float:
        """Simple perplexity estimation based on token frequency"""
        if not tokens:
            return float('inf')
            
        # Count token frequencies
        freq = Counter(tokens)
        total = len(tokens)
        
        # Calculate entropy
        entropy = 0.0
        for token, count in freq.items():
            p = count / total
            entropy -= p * math.log(p + 1e-10)  # Add small epsilon to avoid log(0)
            
        # Perplexity = e^entropy
        return math.exp(entropy)
        
# ============================================================================
# CORE INTELLIGENCE ENGINE
# ============================================================================
class VoxelPathProductionEngine:
    """Production-grade stochastic graph-based architecture with training metrics and optimization"""
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
        self.max_batch_size = 100000  # Maximum tokens per batch
        
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
        self.geometric_momentum = GeometricMomentum(position=(self.resolution//2, self.resolution//2, self.resolution//2))
        self.recent_context = deque(maxlen=50)  # For repetition penalty and coherence
        
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
        """Map token to 3D coordinate using intelligent hashing with bounds checking"""
        if not token or len(token.strip()) == 0:
            # Return center position for empty tokens
            return (self.resolution//2, self.resolution//2, self.resolution//2)
            
        # Use first three characters or pad
        padded = (token + '   ')[:3]
        
        # Convert to ASCII values with bounds checking
        x = min(max(0, ord(padded[0]) % self.resolution), self.resolution - 1)
        y = min(max(0, ord(padded[1]) % self.resolution), self.resolution - 1)
        z = min(max(0, ord(padded[2]) % self.resolution), self.resolution - 1)
        
        return (int(x), int(y), int(z))
        
    def train_corpus(self, text: str, batch_size: int = 10000, validation_text: str = None, progress_callback: Callable = None):
        """Advanced training with hierarchical N-Gram learning and validation metrics"""
        start_time = time.time()
        print(f"[TRAIN] Starting training on {len(text):,} characters")
        
        # Tokenize corpus with progress reporting
        print("[TRAIN] Tokenizing corpus...")
        tokens = self.tokenizer.tokenize(text)
        self.tokens_processed += len(tokens)
        print(f"[TRAIN] Tokenized {len(tokens):,} tokens")
        
        # Split for validation if needed
        validation_tokens = []
        if validation_text:
            validation_tokens = self.tokenizer.tokenize(validation_text)
            print(f"[TRAIN] Loaded validation set with {len(validation_tokens):,} tokens")
            
        # Initialize metrics
        current_metrics = TrainingMetrics()
        current_metrics.tokens_processed = self.tokens_processed
        current_metrics.timestamp = datetime.datetime.now()
        
        # Process in batches with dynamic batch sizing
        total_batches = max(1, (len(tokens) // batch_size) + 1)
        tokens_per_second = 0
        
        # Determine optimal batch size based on corpus size
        optimal_batch_size = min(batch_size, max(1000, len(tokens) // 100))
        print(f"[TRAIN] Using batch size: {optimal_batch_size:,}")
        
        for batch_idx in range(total_batches):
            if progress_callback and not progress_callback(batch_idx, total_batches):
                print("[TRAIN] Training cancelled by user")
                return current_metrics
                
            batch_start = batch_idx * optimal_batch_size
            batch_end = min(batch_start + optimal_batch_size, len(tokens))
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
            self._process_batch(batch_tokens)
            batch_time = time.time() - batch_start_time
            
            # Update metrics
            current_metrics.epoch = batch_idx + 1
            current_metrics.tokens_processed = self.tokens_processed
            current_metrics.ngrams_learned = {n: len(self.ngram_memory[level]) 
                                              for n, level in enumerate(NGramLevel, start=2) 
                                              if level in self.ngram_memory}
            
            # Calculate perplexity on validation set if available
            if validation_tokens and len(validation_tokens) > 100:
                current_metrics.perplexity = self._calculate_validation_perplexity(validation_tokens[:200])
                current_metrics.validation_loss = current_metrics.perplexity
                
            # Memory optimization every 5 batches or if memory is high
            if (batch_idx % 5 == 0 and batch_idx > 0) or self.memory_usage_mb > 2000:
                print(f"[TRAIN] Batch {batch_idx+1}/{total_batches}: Optimizing memory (current usage: {self.memory_usage_mb:.1f} MB)")
                self._optimize_memory()
                current_metrics.memory_usage_mb = self.memory_usage_mb
                
            # Text quality metrics
            if batch_tokens:
                analysis = self.tokenizer.analyze_text(batch_tokens[:min(100, len(batch_tokens))])
                current_metrics.coherence_score = analysis.coherence_score
                current_metrics.diversity_score = analysis.diversity_score
                
            # Save metrics history
            self.training_history.append(current_metrics.to_dict())
            
            # Print progress
            progress = (batch_idx + 1) / total_batches * 100
            print(f"[TRAIN] Batch {batch_idx+1}/{total_batches} ({progress:.1f}%): "
                  f"{len(batch_tokens)} tokens, {tokens_per_second:.0f} tokens/sec, "
                  f"Perplexity: {current_metrics.perplexity:.2f}, "
                  f"Time: {batch_time:.2f}s")
            
            # Create new metrics object for next batch
            current_metrics = TrainingMetrics()
            current_metrics.tokens_processed = self.tokens_processed
            current_metrics.timestamp = datetime.datetime.now()
            
        # Final optimization
        print("[TRAIN] Performing final memory optimization...")
        self._optimize_memory()
        
        # Final metrics
        final_metrics = TrainingMetrics()
        final_metrics.epoch = total_batches
        final_metrics.tokens_processed = self.tokens_processed
        final_metrics.ngrams_learned = {n: len(self.ngram_memory[level]) 
                                       for n, level in enumerate(NGramLevel, start=2) 
                                       if level in self.ngram_memory}
        final_metrics.memory_usage_mb = self.memory_usage_mb
        final_metrics.training_rate = tokens_per_second
        
        if validation_tokens:
            final_metrics.perplexity = self._calculate_validation_perplexity(validation_tokens)
            final_metrics.validation_loss = final_metrics.perplexity
            
        self.training_history.append(final_metrics.to_dict())
        self.training_time += time.time() - start_time
        
        print(f"[TRAIN] Training complete. Learned {sum(final_metrics.ngrams_learned.values()):,} N-Grams. "
              f"Final perplexity: {final_metrics.perplexity:.2f}, "
              f"Training time: {self.training_time:.1f}s")
              
        return final_metrics
        
    def _process_batch(self, tokens: List[str]):
        """Process a single batch of tokens for training with progress reporting"""
        # Train N-Grams from 2 to 5
        for n in range(2, 6):
            ngram_level = NGramLevel(n)
            for i in range(len(tokens) - n + 1):
                context = tuple(tokens[i:i+n-1])
                next_word = tokens[i+n-1]
                
                # Get or create node
                node = self.ngram_memory[ngram_level][context]
                node.add_transition(next_word, weight=1.0)
                node.context = context
                
                # Map to 3D space
                pos = self._calculate_3d_position(next_word)
                node.vector_position = pos
                
                # Update voxel density
                if (0 <= pos[0] < self.resolution and 
                    0 <= pos[1] < self.resolution and 
                    0 <= pos[2] < self.resolution):
                    self.voxel_tensor[pos] += 1.0
                    
                # Update spatial graph
                if node.vector_position and i > 0:
                    prev_context = context[:-1] if len(context) > 1 else ()
                    if prev_context:
                        prev_node = self.ngram_memory[ngram_level].get(prev_context)
                        if prev_node and prev_node.vector_position:
                            self.spatial_graph[prev_node.vector_position].add(node.vector_position)
                            
        # Update N-Grams learned count
        for n in range(2, 6):
            ngram_level = NGramLevel(n)
            self.ngrams_learned[n] = len(self.ngram_memory[ngram_level])
            
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
        Dynamic repetition penalty based on recent context with semantic awareness
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
                # Calculate semantic similarity
                similarity = self._calculate_semantic_similarity(word, last_words)
                if similarity < 0.3:  # Low similarity
                    penalized_probs[word] *= 0.8
                    
        # Renormalize
        total = sum(penalized_probs.values())
        if total > 0:
            return {w: p / total for w, p in penalized_probs.items()}
        return probabilities
        
    def _calculate_semantic_similarity(self, word: str, context_words: List[str]) -> float:
        """Calculate simple semantic similarity between a word and context"""
        if not word or not context_words:
            return 0.0
            
        # Simple prefix and suffix matching
        prefix_sim = sum(1 for w in context_words if w.startswith(word[:2])) / len(context_words)
        suffix_sim = sum(1 for w in context_words if w.endswith(word[-2:])) / len(context_words)
        
        # Length similarity
        len_sim = 1.0 - abs(len(word) - sum(len(w) for w in context_words)/len(context_words)) / max(len(word), sum(len(w) for w in context_words)/len(context_words))
        
        return (prefix_sim + suffix_sim + len_sim) / 3.0
        
    def a_star_search(self,
                     start_pos: Tuple[int, int, int],
                     target_chars: Optional[List[str]] = None,
                     max_iterations: int = 1000) -> List[Tuple[int, int, int]]:
        """
        A* pathfinding through voxel space with boundary checking and timeout
        """
        start_time = time.time()
        
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
            # Check for timeout
            if time.time() - start_time > 5.0:  # 5 second timeout
                print("[A*] Pathfinding timeout, returning partial path")
                break
                
            f_score, g_score, current, path = heapq.heappop(open_set)
            iterations += 1
            
            # Check if reached target
            if current in target_positions:
                return path
                
            # Explore neighbors from spatial graph
            neighbors = self.spatial_graph.get(current, set())
            
            # Add nearby voxels based on voxel density if no neighbors found
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
                            dist = math.sqrt(
                                (neighbor[0]-t[0])**2 + 
                                (neighbor[1]-t[1])**2 + 
                                (neighbor[2]-t[2])**2
                            )
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
        
    def generate_text(self,
                     seed_text: str,
                     max_length: int = 200,
                     temperature: float = 0.85,
                     top_p: float = 0.9,
                     repetition_penalty: float = 1.2,
                     use_astar: bool = False,
                     use_momentum: bool = True,
                     min_length: int = 50) -> Tuple[str, List[Tuple[int, int, int]], TextAnalysisResult]:
        """
        Advanced text generation with all features enabled and comprehensive metrics
        """
        start_time = time.time()
        print(f"[GENERATE] Starting generation with seed: '{seed_text[:50]}...'")
        
        # Tokenize seed
        seed_tokens = self.tokenizer.tokenize(seed_text)
        if len(seed_tokens) < 1:
            # Default to classic literary start
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
            if not probabilities:
                # Try simpler context
                if len(context) > 1:
                    probabilities = self.recursive_backoff(context[-1:])
                    
                # Still nothing? Use most common words
                if not probabilities:
                    # Get most frequent words from bigrams
                    bigram_nodes = list(self.ngram_memory[NGramLevel.BIGRAM].values())
                    if bigram_nodes:
                        # Get the most frequent word across all nodes
                        word_freq = Counter()
                        for node in bigram_nodes:
                            for word, count in node.next_words.items():
                                word_freq[word] += count
                                
                        most_common = word_freq.most_common(10)
                        probabilities = {word: count/sum(c for _, c in most_common) 
                                       for word, count in most_common}
                    else:
                        # Ultimate fallback
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
                    scaled = math.exp(math.log(prob) / temperature) if prob > 0 else 0
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
                
            # Check for natural ending points
            if i >= min_length - 1 and next_word in ['.', '!', '?', '\n']:
                break
                
        # Convert tokens back to text
        generated_text = self.tokenizer.detokenize(generated_tokens)
        
        # Analyze the generated text
        analysis_metrics = self.tokenizer.analyze_text(generated_tokens)
        generation_time = time.time() - start_time
        
        print(f"[GENERATE] Completed in {generation_time:.2f}s. Generated {len(generated_text):,} characters")
        print(f"[GENERATE] Quality metrics - Coherence: {analysis_metrics.coherence_score:.2f}, "
              f"Diversity: {analysis_metrics.diversity_score:.2f}, "
              f"Perplexity: {analysis_metrics.perplexity:.2f}")
        
        return generated_text, trajectory, analysis_metrics
        
    def _optimize_memory(self):
        """Memory optimization and cleanup with metrics tracking and progress reporting"""
        print("[OPTIMIZE] Running memory optimization...")
        start_time = time.time()
        
        # Prune low-frequency N-Grams
        total_pruned = 0
        for level_name, level in self.ngram_memory.items():
            to_delete = []
            for key, node in level.items():
                # Keep nodes that have been seen multiple times or recently updated
                if node.total_count < 2 and (datetime.datetime.now() - node.last_updated).total_seconds() > 300:  # 5 minutes
                    to_delete.append(key)
                    
            total_pruned += len(to_delete)
            for key in to_delete:
                del level[key]
                
        # Clear spatial graph for pruned nodes and rebuild
        self.spatial_graph.clear()
        
        # Rebuild spatial graph with remaining nodes
        connections_added = 0
        nodes_processed = 0
        total_nodes = sum(len(nodes) for nodes in self.ngram_memory.values())
        
        for level_name, nodes in self.ngram_memory.items():
            for node in nodes.values():
                nodes_processed += 1
                if not node.vector_position:
                    continue
                    
                # Only connect to nearby nodes
                nearby_nodes = []
                for other_node in nodes.values():
                    if other_node == node or not other_node.vector_position:
                        continue
                        
                    # Calculate distance
                    try:
                        dist = math.sqrt(
                            (node.vector_position[0] - other_node.vector_position[0])**2 +
                            (node.vector_position[1] - other_node.vector_position[1])**2 +
                            (node.vector_position[2] - other_node.vector_position[2])**2
                        )
                        if dist < 30:  # Connection threshold
                            nearby_nodes.append((dist, other_node))
                    except Exception:
                        continue
                        
                # Connect to top 5 closest nodes
                nearby_nodes.sort(key=lambda x: x[0])
                for dist, other_node in nearby_nodes[:5]:
                    self.spatial_graph[node.vector_position].add(other_node.vector_position)
                    connections_added += 1
                    
                # Progress reporting for large graphs
                if nodes_processed % 1000 == 0:
                    progress = (nodes_processed / total_nodes) * 100
                    print(f"[OPTIMIZE] Progress: {progress:.1f}% ({nodes_processed:,}/{total_nodes:,} nodes)")
                    
        # Prune old trajectory tunnels
        tunnel_keys = list(self.trajectory_tunnels.keys())
        tunnels_pruned = 0
        for key in tunnel_keys:
            tunnel = self.trajectory_tunnels[key]
            if tunnel.usage_count < 3 and (datetime.datetime.now() - tunnel.creation_time).total_seconds() > 3600:  # 1 hour
                del self.trajectory_tunnels[key]
                tunnels_pruned += 1
                
        # Update memory usage estimate
        self._update_memory_stats()
        self.last_optimization = datetime.datetime.now()
        
        optimization_time = time.time() - start_time
        print(f"[OPTIMIZE] Memory optimized in {optimization_time:.2f}s. "
              f"Pruned {total_pruned:,} N-Grams, {tunnels_pruned} tunnels. "
              f"Added {connections_added:,} spatial connections. "
              f"Current usage: {self.memory_usage_mb:.1f} MB")
              
    def _update_memory_stats(self):
        """Calculate current memory usage accurately"""
        # Estimate N-Gram memory
        ngram_memory = 0
        for level, nodes in self.ngram_memory.items():
            for node in nodes.values():
                # Context memory
                if node.context:
                    ngram_memory += sum(len(str(token)) for token in node.context) * 2
                
                # Next words memory
                for word, count in node.next_words.items():
                    ngram_memory += len(str(word)) * 2 + 8  # 8 bytes for float
                    
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
            'model_version': '3.2'
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
                'ngrams_learned': {str(n): 0 for n in range(2, 6)},
                'resolution': self.resolution,
                'memory_usage_mb': 0,
                'training_time': 0
            }
            
        # Load N-Gram memory
        ngram_path = os.path.join(model_path, "ngram_memory.pkl.gz")
        try:
            with gzip.open(ngram_path, 'rb') as f:
                loaded_ngrams = pickle.load(f)
                
            # Convert back to defaultdict with proper types
            self.ngram_memory.clear()
            for level_str, nodes in loaded_ngrams.items():
                # Handle both string and enum keys
                if isinstance(level_str, str):
                    try:
                        level = NGramLevel[level_str]
                    except KeyError:
                        # Try numeric conversion
                        try:
                            level_value = int(level_str.split('.')[-1])
                            level = NGramLevel(level_value)
                        except:
                            continue
                else:
                    level = level_str
                    
                self.ngram_memory[level] = defaultdict(
                    lambda: NGramNode(context=()), 
                    nodes
                )
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
        
# ============================================================================
# ASYNCHRONOUS WORKERS
# ============================================================================
class TrainingWorker(QThread):
    """Advanced training worker with real-time progress reporting and validation"""
    # Signals
    progress_updated = pyqtSignal(int, str, dict)  # (progress_percent, status_message, metrics)
    training_complete = pyqtSignal(str, dict)  # (completion_message, final_metrics)
    error_occurred = pyqtSignal(str)  # error_message
    validation_metrics = pyqtSignal(dict)  # validation metrics
    
    def __init__(self, engine: VoxelPathProductionEngine, corpus_path: str, validation_path: str = None):
        super().__init__()
        self.engine = engine
        self.corpus_path = corpus_path
        self.validation_path = validation_path
        self.is_running = True
        self.total_chars = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.batch_size = 50000
        
    def run(self):
        """Main training execution with validation support"""
        try:
            # Read corpus
            self.progress_updated.emit(0, f"Reading corpus: {os.path.basename(self.corpus_path)}", {})
            with open(self.corpus_path, 'r', encoding='utf-8', errors='replace') as f:
                corpus_text = f.read()
                
            # Read validation corpus if available
            validation_text = None
            if self.validation_path and os.path.exists(self.validation_path):
                self.progress_updated.emit(2, "Reading validation corpus...", {})
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
            total_batches = max(1, (file_size // self.batch_size) + 1)
            
            # Process in chunks
            metrics_history = []
            
            for batch_idx in range(total_batches):
                if not self.is_running:
                    break
                    
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, file_size)
                batch_text = corpus_text[batch_start:batch_end]
                
                # Train on batch
                batch_start_time = time.time()
                metrics = self.engine.train_corpus(
                    batch_text, 
                    batch_size=10000,
                    validation_text=validation_text,
                    progress_callback=self._check_cancel
                )
                batch_time = time.time() - batch_start_time
                
                # Update progress
                progress = min(95, int(((batch_idx + 1) / total_batches) * 90) + 5)
                elapsed = time.time() - self.start_time
                eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1) if batch_idx > 0 else 0
                
                status = f"Batch {batch_idx+1}/{total_batches}: {len(batch_text):,} chars, "
                status += f"{metrics.training_rate:.0f} tokens/sec, "
                status += f"Perplexity: {metrics.perplexity:.2f}, "
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
                
                # Periodic validation every 3 batches
                if validation_text and (batch_idx + 1) % 3 == 0:
                    self._run_validation(validation_text)
                    
                # Memory optimization every 10 batches
                if (batch_idx + 1) % 10 == 0 and batch_idx > 0:
                    self.progress_updated.emit(progress, "Optimizing memory...", {})
                    
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
            error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
            
    def _check_cancel(self, batch_idx: int, total_batches: int) -> bool:
        """Callback to check if training should be cancelled"""
        if not self.is_running:
            return False
            
        # Update progress every 10 batches or every 2 seconds
        current_time = time.time()
        if batch_idx % 10 == 0 or current_time - self.last_update_time > 2.0:
            progress = min(95, int((batch_idx / total_batches) * 90) + 5)
            self.progress_updated.emit(progress, f"Processing batch {batch_idx+1}/{total_batches}...", {})
            self.last_update_time = current_time
            
        return True
        
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
        self.is_running = True
        
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
            
    def stop(self):
        """Stop generation gracefully"""
        self.is_running = False
        
# ============================================================================
# ANALYSIS AND VISUALIZATION COMPONENTS
# ============================================================================
class TrainingMetricsTableModel(QAbstractTableModel):
    """Model for displaying training metrics in a table"""
    def __init__(self, metrics_history=None, parent=None):
        super().__init__(parent)
        self.metrics_history = metrics_history or []
        self.headers = [
            'Batch', 'Tokens', '2-grams', '3-grams', '4-grams', '5-grams', 
            'Perplexity', 'Coherence', 'Diversity', 'Memory (MB)', 'Rate (tok/s)'
        ]
        
    def rowCount(self, parent=None):
        return len(self.metrics_history)
        
    def columnCount(self, parent=None):
        return len(self.headers)
        
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
            
        row = index.row()
        col = index.column()
        metrics = self.metrics_history[row]
        
        try:
            if col == 0:  # Batch
                return str(metrics.get('batch_idx', row) + 1)
            elif col == 1:  # Tokens
                return f"{metrics.get('tokens_processed', 0):,}"
            elif col == 2:  # 2-grams
                return f"{metrics.get('ngrams_learned', {}).get(2, 0):,}"
            elif col == 3:  # 3-grams
                return f"{metrics.get('ngrams_learned', {}).get(3, 0):,}"
            elif col == 4:  # 4-grams
                return f"{metrics.get('ngrams_learned', {}).get(4, 0):,}"
            elif col == 5:  # 5-grams
                return f"{metrics.get('ngrams_learned', {}).get(5, 0):,}"
            elif col == 6:  # Perplexity
                return f"{metrics.get('perplexity', float('inf')):.2f}"
            elif col == 7:  # Coherence
                return f"{metrics.get('coherence_score', 0.0):.2f}"
            elif col == 8:  # Diversity
                return f"{metrics.get('diversity_score', 0.0):.2f}"
            elif col == 9:  # Memory
                return f"{metrics.get('memory_usage_mb', 0.0):.1f}"
            elif col == 10:  # Rate
                return f"{metrics.get('training_rate', 0.0):.0f}"
        except (KeyError, TypeError, ValueError):
            return "N/A"
            
        return None
        
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return super().headerData(section, orientation, role)
        
class TrainingProgressVisualizer(QWidget):
    """Widget for visualizing training progress with matplotlib"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Create figure and axes
        self.fig = Figure(figsize=(10, 6), facecolor='#2a2a3a')
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(221, facecolor='#2a2a3a')  # Perplexity
        self.ax2 = self.fig.add_subplot(222, facecolor='#2a2a3a')  # N-Grams
        self.ax3 = self.fig.add_subplot(223, facecolor='#2a2a3a')  # Memory/Rate
        self.ax4 = self.fig.add_subplot(224, facecolor='#2a2a3a')  # Coherence/Diversity
        
        # Style the plots
        self._style_plots()
        
        # Data storage
        self.metrics_history = []
        
    def _style_plots(self):
        """Apply consistent styling to all plots"""
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.grid(True, alpha=0.3, color='#555555')
            ax.tick_params(colors='#cccccc')
            for spine in ax.spines.values():
                spine.set_color('#555555')
            ax.xaxis.label.set_color('#cccccc')
            ax.yaxis.label.set_color('#cccccc')
            ax.title.set_color('#cccccc')
            
    def update_metrics(self, metrics_history):
        """Update visualizations with new metrics data"""
        self.metrics_history = metrics_history
        self._update_plots()
        
    def _update_plots(self):
        """Update all plot data"""
        if not self.metrics_history:
            return
            
        # Extract data
        batches = [m.get('batch_idx', i) for i, m in enumerate(self.metrics_history)]
        perplexities = [m.get('perplexity', float('inf')) for m in self.metrics_history]
        coherence_scores = [m.get('coherence_score', 0.0) for m in self.metrics_history]
        diversity_scores = [m.get('diversity_score', 0.0) for m in self.metrics_history]
        memory_usage = [m.get('memory_usage_mb', 0.0) for m in self.metrics_history]
        training_rates = [m.get('training_rate', 0.0) for m in self.metrics_history]
        
        # N-Gram counts
        bigrams = [m.get('ngrams_learned', {}).get(2, 0) for m in self.metrics_history]
        trigrams = [m.get('ngrams_learned', {}).get(3, 0) for m in self.metrics_history]
        quadgrams = [m.get('ngrams_learned', {}).get(4, 0) for m in self.metrics_history]
        quintgrams = [m.get('ngrams_learned', {}).get(5, 0) for m in self.metrics_history]
        
        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Plot 1: Perplexity
        self.ax1.plot(batches, perplexities, 'r-', linewidth=2, label='Perplexity')
        self.ax1.set_title('Model Perplexity', fontsize=10)
        self.ax1.set_xlabel('Batch')
        self.ax1.set_ylabel('Perplexity')
        self.ax1.legend(fontsize=8)
        
        # Plot 2: N-Gram counts
        self.ax2.plot(batches, bigrams, 'b-', linewidth=1, label='2-grams')
        self.ax2.plot(batches, trigrams, 'g-', linewidth=1, label='3-grams')
        self.ax2.plot(batches, quadgrams, 'y-', linewidth=1, label='4-grams')
        self.ax2.plot(batches, quintgrams, 'm-', linewidth=1, label='5-grams')
        self.ax2.set_title('N-Gram Learning Progress', fontsize=10)
        self.ax2.set_xlabel('Batch')
        self.ax2.set_ylabel('Count')
        self.ax2.legend(fontsize=6)
        
        # Plot 3: Memory and Training Rate
        ax3b = self.ax3.twinx()
        self.ax3.plot(batches, memory_usage, 'c-', linewidth=2, label='Memory')
        ax3b.plot(batches, training_rates, 'y-', linewidth=2, label='Rate')
        self.ax3.set_title('Resource Usage', fontsize=10)
        self.ax3.set_xlabel('Batch')
        self.ax3.set_ylabel('Memory (MB)', color='c')
        ax3b.set_ylabel('Tokens/sec', color='y')
        self.ax3.tick_params(axis='y', colors='c')
        ax3b.tick_params(axis='y', colors='y')
        
        # Plot 4: Quality Metrics
        self.ax4.plot(batches, coherence_scores, 'g-', linewidth=2, label='Coherence')
        self.ax4.plot(batches, diversity_scores, 'b-', linewidth=2, label='Diversity')
        self.ax4.set_title('Text Quality Metrics', fontsize=10)
        self.ax4.set_xlabel('Batch')
        self.ax4.set_ylabel('Score')
        self.ax4.legend(fontsize=8)
        
        # Apply styling
        self._style_plots()
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Redraw
        self.canvas.draw()
        
    def clear(self):
        """Clear all plots"""
        self.metrics_history = []
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        self.canvas.draw()
        
# ============================================================================
# ANALYSIS DIALOGS
# ============================================================================
class TextAnalysisDialog(QDialog):
    """Dialog for detailed text analysis with advanced metrics"""
    def __init__(self, metrics: dict, text_sample: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Text Analysis Report")
        self.setMinimumSize(700, 600)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Comprehensive Text Quality Analysis")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Tab widget for different analysis sections
        tab_widget = QTabWidget()
        
        # Tab 1: Quality Metrics
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        # Metrics table
        table = QTableWidget(10, 2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        metrics_data = [
            ("Perplexity", f"{metrics.get('perplexity', float('inf')):.2f}"),
            ("Coherence Score", f"{metrics.get('coherence_score', 0.0):.2f}"),
            ("Diversity Score", f"{metrics.get('diversity_score', 0.0):.2f}"),
            ("Topic Consistency", f"{metrics.get('topic_consistency', 0.0):.2f}"),
            ("Grammatical Correctness", f"{metrics.get('grammatical_correctness', 0.0):.2f}"),
            ("Repetition Rate", f"{metrics.get('repetition_rate', 0.0):.2f}"),
            ("Avg Word Length", f"{metrics.get('avg_word_length', 0.0):.2f}"),
            ("Sentence Complexity", f"{metrics.get('sentence_complexity', 0.0):.2f}"),
            ("Generation Time", f"{metrics.get('generation_time', 0.0):.2f} seconds"),
            ("Character Count", f"{metrics.get('character_count', 0)}")
        ]
        
        for i, (metric, value) in enumerate(metrics_data):
            table.setItem(i, 0, QTableWidgetItem(metric))
            table.setItem(i, 1, QTableWidgetItem(value))
            
        metrics_layout.addWidget(table)
        
        # Quality gauge visualization
        gauge_layout = QHBoxLayout()
        
        coherence = metrics.get('coherence_score', 0.0)
        diversity = metrics.get('diversity_score', 0.0)
        grammar = metrics.get('grammatical_correctness', 0.0)
        
        # Create simple gauge visualizations
        def create_gauge(label, value, color):
            container = QVBoxLayout()
            lbl = QLabel(f"{label}: {value:.2f}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container.addWidget(lbl)
            
            # Gauge bar
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(value * 100))
            
            if value < 0.3:
                bar.setStyleSheet("QProgressBar::chunk { background-color: #ff4444; }")
            elif value < 0.7:
                bar.setStyleSheet("QProgressBar::chunk { background-color: #ffaa00; }")
            else:
                bar.setStyleSheet("QProgressBar::chunk { background-color: #44ff44; }")
                
            container.addWidget(bar)
            return container
            
        gauge_layout.addLayout(create_gauge("Coherence", coherence, "#44ff44"))
        gauge_layout.addLayout(create_gauge("Diversity", diversity, "#4444ff"))
        gauge_layout.addLayout(create_gauge("Grammar", grammar, "#ff4444"))
        
        metrics_layout.addLayout(gauge_layout)
        
        tab_widget.addTab(metrics_tab, "Quality Metrics")
        
        # Tab 2: Text Sample
        sample_tab = QWidget()
        sample_layout = QVBoxLayout(sample_tab)
        
        sample_label = QLabel("Generated Text Sample:")
        sample_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        sample_layout.addWidget(sample_label)
        
        sample_text = QTextEdit()
        sample_text.setReadOnly(True)
        sample_text.setFont(QFont("Georgia", 11))
        sample_text.setStyleSheet("""
        QTextEdit {
            background-color: #1e1e2e;
            color: #f0f0f0;
            border: 2px solid #333344;
            padding: 10px;
            line-height: 1.4;
        }
        """)
        sample_text.setPlainText(text_sample)
        sample_layout.addWidget(sample_text)
        
        tab_widget.addTab(sample_tab, "Text Sample")
        
        # Tab 3: Recommendations
        recommendations_tab = QWidget()
        rec_layout = QVBoxLayout(recommendations_tab)
        
        recommendations = self._generate_recommendations(metrics)
        
        rec_label = QLabel("Optimization Recommendations:")
        rec_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        rec_layout.addWidget(rec_label)
        
        rec_text = QTextEdit()
        rec_text.setReadOnly(True)
        rec_text.setFont(QFont("Consolas", 10))
        rec_text.setStyleSheet("""
        QTextEdit {
            background-color: #2a2a3a;
            color: #aaffaa;
            border: 1px solid #444444;
            padding: 10px;
        }
        """)
        rec_text.setPlainText(recommendations)
        rec_layout.addWidget(rec_text)
        
        tab_widget.addTab(recommendations_tab, "Recommendations")
        
        layout.addWidget(tab_widget)
        
        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    def _generate_recommendations(self, metrics: dict) -> str:
        """Generate optimization recommendations based on metrics"""
        recs = []
        recs.append("MODEL OPTIMIZATION RECOMMENDATIONS")
        recs.append("=" * 50)
        recs.append("")
        
        # Analyze metrics
        coherence = metrics.get('coherence_score', 0.0)
        diversity = metrics.get('diversity_score', 0.0)
        grammar = metrics.get('grammatical_correctness', 0.0)
        repetition = metrics.get('repetition_rate', 0.0)
        perplexity = metrics.get('perplexity', float('inf'))
        
        # Coherence recommendations
        if coherence < 0.4:
            recs.append(f"LOW COHERENCE ({coherence:.2f}):")
            recs.append("- Increase context window size")
            recs.append("- Lower temperature (0.6-0.7)")
            recs.append("- Increase repetition penalty (1.3-1.5)")
            recs.append("- Train on more coherent text sources")
            recs.append("")
            
        # Diversity recommendations
        if diversity < 0.3:
            recs.append(f"LOW DIVERSITY ({diversity:.2f}):")
            recs.append("- Increase temperature (0.9-1.2)")
            recs.append("- Decrease top-p threshold (0.7-0.8)")
            recs.append("- Add more varied training data")
            recs.append("- Reduce repetition penalty (0.9-1.1)")
            recs.append("")
            
        # Grammar recommendations
        if grammar < 0.6:
            recs.append(f"POOR GRAMMAR ({grammar:.2f}):")
            recs.append("- Train on more grammatically correct text")
            recs.append("- Increase bigram/trigram weight")
            recs.append("- Add explicit grammar rules")
            recs.append("- Use lower temperature (0.5-0.7)")
            recs.append("")
            
        # Repetition recommendations
        if repetition > 0.3:
            recs.append(f"HIGH REPETITION ({repetition:.2f}):")
            recs.append("- Increase repetition penalty (1.4-2.0)")
            recs.append("- Increase temperature (0.9-1.1)")
            recs.append("- Use more aggressive pruning")
            recs.append("")
            
        # Perplexity recommendations
        if perplexity > 100:
            recs.append(f"HIGH PERPLEXITY ({perplexity:.2f}):")
            recs.append("- More training data needed")
            recs.append("- Longer training time")
            recs.append("- Check data quality and preprocessing")
            recs.append("")
            
        # General recommendations
        recs.append("GENERAL OPTIMIZATION STRATEGIES:")
        recs.append("- Use validation set for early stopping")
        recs.append("- Monitor metrics over multiple generations")
        recs.append("- Experiment with different context lengths")
        recs.append("- Try different sampling strategies")
        recs.append("- Consider fine-tuning on domain-specific data")
        
        if not recs[2:]:  # If no specific recommendations
            recs.append("\nEXCELLENT METRICS! No major optimizations needed.")
            recs.append("Continue monitoring and consider:")
            recs.append("- Increasing generation length")
            recs.append("- Exploring different creative modes")
            recs.append("- Testing with more challenging prompts")
            
        return "\n".join(recs)
        
class BatchGenerationDialog(QDialog):
    """Dialog for batch text generation with parameter variation"""
    def __init__(self, engine: VoxelPathProductionEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setWindowTitle("Advanced Batch Text Generation")
        self.setMinimumSize(700, 500)
        
        layout = QVBoxLayout()
        
        # Seed text with examples
        seed_group = QGroupBox("Seed Text (select example or enter custom)")
        seed_layout = QVBoxLayout()
        
        # Example seeds
        self.example_seeds = {
            "Classic Literature": "it was the best of times, it was the worst of times",
            "Philosophical": "to be or not to be, that is the question",
            "Mystery": "the door opened slowly, revealing a room bathed in shadows",
            "Sci-Fi": "the stars stretched endlessly before the spacecraft",
            "Fantasy": "once upon a time, in a kingdom far away",
            "Custom": ""
        }
        
        self.combo_seed_examples = QComboBox()
        self.combo_seed_examples.addItems(self.example_seeds.keys())
        self.combo_seed_examples.currentTextChanged.connect(self.update_seed_text)
        seed_layout.addWidget(self.combo_seed_examples)
        
        self.txt_seed = QTextEdit()
        self.txt_seed.setPlaceholderText("Enter seed text for batch generation")
        self.txt_seed.setMaximumHeight(100)
        seed_layout.addWidget(self.txt_seed)
        
        seed_group.setLayout(seed_layout)
        layout.addWidget(seed_group)
        
        # Parameters with ranges
        param_group = QGroupBox("Generation Parameters (range for variation)")
        param_layout = QGridLayout()
        
        # Number of texts
        param_layout.addWidget(QLabel("Number of texts:"), 0, 0)
        self.spin_num_texts = QSpinBox()
        self.spin_num_texts.setRange(1, 50)
        self.spin_num_texts.setValue(10)
        param_layout.addWidget(self.spin_num_texts, 0, 1)
        
        # Length range
        param_layout.addWidget(QLabel("Length range:"), 1, 0)
        length_layout = QHBoxLayout()
        self.spin_min_length = QSpinBox()
        self.spin_min_length.setRange(50, 2000)
        self.spin_min_length.setValue(100)
        self.spin_max_length = QSpinBox()
        self.spin_max_length.setRange(50, 2000)
        self.spin_max_length.setValue(300)
        length_layout.addWidget(self.spin_min_length)
        length_layout.addWidget(QLabel("to"))
        length_layout.addWidget(self.spin_max_length)
        param_layout.addLayout(length_layout, 1, 1)
        
        # Temperature range
        param_layout.addWidget(QLabel("Temperature range:"), 2, 0)
        temp_layout = QHBoxLayout()
        self.spin_min_temp = QDoubleSpinBox()
        self.spin_min_temp.setRange(0.1, 2.0)
        self.spin_min_temp.setValue(0.7)
        self.spin_max_temp = QDoubleSpinBox()
        self.spin_max_temp.setRange(0.1, 2.0)
        self.spin_max_temp.setValue(1.2)
        temp_layout.addWidget(self.spin_min_temp)
        temp_layout.addWidget(QLabel("to"))
        temp_layout.addWidget(self.spin_max_temp)
        param_layout.addLayout(temp_layout, 2, 1)
        
        # Top-P range
        param_layout.addWidget(QLabel("Top-P range:"), 3, 0)
        top_p_layout = QHBoxLayout()
        self.spin_min_top_p = QDoubleSpinBox()
        self.spin_min_top_p.setRange(0.1, 1.0)
        self.spin_min_top_p.setValue(0.8)
        self.spin_max_top_p = QDoubleSpinBox()
        self.spin_max_top_p.setRange(0.1, 1.0)
        self.spin_max_top_p.setValue(0.95)
        top_p_layout.addWidget(self.spin_min_top_p)
        top_p_layout.addWidget(QLabel("to"))
        top_p_layout.addWidget(self.spin_max_top_p)
        param_layout.addLayout(top_p_layout, 3, 1)
        
        # Repetition penalty range
        param_layout.addWidget(QLabel("Repetition penalty range:"), 4, 0)
        rep_layout = QHBoxLayout()
        self.spin_min_rep = QDoubleSpinBox()
        self.spin_min_rep.setRange(0.5, 3.0)
        self.spin_min_rep.setValue(1.0)
        self.spin_max_rep = QDoubleSpinBox()
        self.spin_max_rep.setRange(0.5, 3.0)
        self.spin_max_rep.setValue(1.5)
        rep_layout.addWidget(self.spin_min_rep)
        rep_layout.addWidget(QLabel("to"))
        rep_layout.addWidget(self.spin_max_rep)
        param_layout.addLayout(rep_layout, 4, 1)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QGridLayout()
        
        output_layout.addWidget(QLabel("Output directory:"), 0, 0)
        self.txt_output_dir = QLineEdit()
        self.txt_output_dir.setText(os.path.join(os.getcwd(), "generated_texts"))
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.browse_directory)
        output_layout.addWidget(self.txt_output_dir, 0, 1)
        output_layout.addWidget(self.btn_browse, 0, 2)
        
        output_layout.addWidget(QLabel("File naming:"), 1, 0)
        self.combo_naming = QComboBox()
        self.combo_naming.addItems([
            "sequential (text_001.txt)",
            "timestamp (text_20240101_123045.txt)", 
            "metrics (text_coherence_0.85.txt)"
        ])
        output_layout.addWidget(self.combo_naming, 1, 1, 1, 2)
        
        self.cb_analyze = QCheckBox("Include quality analysis in filename")
        self.cb_analyze.setChecked(True)
        output_layout.addWidget(self.cb_analyze, 2, 0, 1, 3)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Progress and controls
        progress_layout = QVBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        progress_layout.addWidget(self.progress)
        
        button_layout = QHBoxLayout()
        self.btn_generate = QPushButton("Start Batch Generation")
        self.btn_generate.clicked.connect(self.start_generation)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_generate)
        button_layout.addWidget(self.btn_cancel)
        progress_layout.addLayout(button_layout)
        
        layout.addLayout(progress_layout)
        
        self.setLayout(layout)
        self.generated_files = []
        
        # Set default seed
        self.update_seed_text("Classic Literature")
        
    def update_seed_text(self, example_name: str):
        """Update seed text when example is selected"""
        seed_text = self.example_seeds.get(example_name, "")
        self.txt_seed.setPlainText(seed_text)
        
    def browse_directory(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Output Directory",
            self.txt_output_dir.text()
        )
        if directory:
            self.txt_output_dir.setText(directory)
            
    def start_generation(self):
        """Start batch generation with progress reporting"""
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
                
        # Get parameters
        num_texts = self.spin_num_texts.value()
        min_length = self.spin_min_length.value()
        max_length = self.spin_max_length.value()
        min_temp = self.spin_min_temp.value()
        max_temp = self.spin_max_temp.value()
        min_top_p = self.spin_min_top_p.value()
        max_top_p = self.spin_max_top_p.value()
        min_rep = self.spin_min_rep.value()
        max_rep = self.spin_max_rep.value()
        
        self.btn_generate.setEnabled(False)
        self.progress.setValue(0)
        self.generated_files = []
        
        # Show progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Batch Generation Progress")
        progress_dialog.setFixedSize(400, 150)
        dialog_layout = QVBoxLayout(progress_dialog)
        
        status_label = QLabel("Starting generation...")
        dialog_layout.addWidget(status_label)
        
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        dialog_layout.addWidget(progress_bar)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(lambda: setattr(self, 'is_cancelled', True))
        dialog_layout.addWidget(btn_cancel)
        
        progress_dialog.show()
        QApplication.processEvents()
        
        self.is_cancelled = False
        
        # Generate texts
        for i in range(num_texts):
            if self.is_cancelled or self.isHidden():  # Dialog was closed or cancelled
                break
                
            # Sample parameters within ranges
            length = random.randint(min_length, max_length)
            temperature = random.uniform(min_temp, max_temp)
            top_p = random.uniform(min_top_p, max_top_p)
            repetition_penalty = random.uniform(min_rep, max_rep)
            
            # Update progress dialog
            progress = int((i / num_texts) * 100)
            status_label.setText(f"Generating text {i+1}/{num_texts}...")
            progress_bar.setValue(progress)
            QApplication.processEvents()
            
            # Generate text
            generated_text, _, metrics = self.engine.generate_text(
                seed_text=seed_text,
                max_length=length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_astar=False,
                use_momentum=True,
                min_length=min_length
            )
            
            # Generate filename based on options
            naming_scheme = self.combo_naming.currentText()
            include_analysis = self.cb_analyze.isChecked()
            
            base_name = "text"
            if "sequential" in naming_scheme:
                filename = f"{base_name}_{i+1:03d}.txt"
            elif "timestamp" in naming_scheme:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{base_name}_{timestamp}_{i+1:03d}.txt"
            else:  # metrics-based
                coherence = metrics.get('coherence_score', 0.0)
                diversity = metrics.get('diversity_score', 0.0)
                filename = f"{base_name}_coh{coherence:.2f}_div{diversity:.2f}_{i+1:03d}.txt"
                
            # Add metrics to filename if requested
            if include_analysis:
                coherence = metrics.get('coherence_score', 0.0)
                diversity = metrics.get('diversity_score', 0.0)
                filename = filename.replace(".txt", f"_coh{coherence:.2f}_div{diversity:.2f}.txt")
                
            # Save to file
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                # Add metadata header
                f.write(f"VoxelPath Generated Text\n")
                f.write(f"========================\n")
                f.write(f"Seed: {seed_text}\n")
                f.write(f"Length: {length} tokens\n")
                f.write(f"Temperature: {temperature:.2f}\n")
                f.write(f"Top-P: {top_p:.2f}\n")
                f.write(f"Repetition Penalty: {repetition_penalty:.2f}\n")
                f.write(f"Coherence Score: {metrics.get('coherence_score', 0.0):.2f}\n")
                f.write(f"Diversity Score: {metrics.get('diversity_score', 0.0):.2f}\n")
                f.write(f"Generation Time: {metrics.get('generation_time', 0.0):.2f}s\n")
                f.write(f"\n{'='*50}\n\n")
                f.write(generated_text)
                
            self.generated_files.append((filepath, metrics))
            
        # Close progress dialog
        progress_dialog.accept()
        
        self.btn_generate.setEnabled(True)
        
        if self.generated_files and not self.is_cancelled:
            # Show summary
            avg_coherence = sum(m.get('coherence_score', 0.0) for _, m in self.generated_files) / len(self.generated_files)
            avg_diversity = sum(m.get('diversity_score', 0.0) for _, m in self.generated_files) / len(self.generated_files)
            
            summary = f"Generated {len(self.generated_files)} texts successfully!\n\n"
            summary += f"Average Coherence: {avg_coherence:.2f}\n"
            summary += f"Average Diversity: {avg_diversity:.2f}\n\n"
            summary += f"Files saved to: {output_dir}"
            
            QMessageBox.information(self, "Generation Complete", summary)
            self.accept()
        elif self.is_cancelled:
            QMessageBox.information(self, "Generation Cancelled", 
                                  f"Generated {len(self.generated_files)} texts before cancellation.")
            
# ============================================================================
# PRODUCTION GUI
# ============================================================================
class VoxelPathProductionUI(QMainWindow):
    """Industrial-grade production interface with real-time monitoring"""
    def __init__(self):
        super().__init__()
        # Set application style
        if QApplication.instance():
            QApplication.instance().setStyleSheet(qdarktheme.load_stylesheet("dark"))
            
        # Core engine
        self.engine = VoxelPathProductionEngine()
        
        # Worker threads
        self.training_worker = None
        self.generation_worker = None
        
        # UI state
        self.current_trajectory = []
        self.loaded_models = []
        self.last_generated_text = ""
        self.last_generation_metrics = {}
        self.training_metrics_history = []
        self.validation_results = []
        
        # Setup UI
        self.setWindowTitle("VoxelPath Production System v3.2")
        self.resize(1600, 900)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create central widget and main layout
        self._setup_ui()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System: Initializing...")
        
        # Initialize system
        self._system_log("System initialized. Ready for production.")
        
        # Update timer for real-time stats
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_status_bar)
        self.stats_timer.start(5000)  # Update every 5 seconds
        
        # Show startup message
        self.status_bar.showMessage("System: Ready | Use File > Load Corpus to begin training")
        
        # Initialize training cancellation flag
        self.training_cancelled = False
        
    def _create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Engine", self)
        new_action.triggered.connect(self.reset_engine)
        file_menu.addAction(new_action)
        
        load_corpus_action = QAction("Load Training Corpus", self)
        load_corpus_action.triggered.connect(self.load_corpus)
        file_menu.addAction(load_corpus_action)
        
        load_val_action = QAction("Load Validation Corpus", self)
        load_val_action.triggered.connect(self.load_validation_corpus)
        file_menu.addAction(load_val_action)
        
        file_menu.addSeparator()
        
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
        
        analyze_action = QAction("Analyze Generated Text", self)
        analyze_action.triggered.connect(self.analyze_text)
        tools_menu.addAction(analyze_action)
        
        compare_action = QAction("Compare Models", self)
        compare_action.triggered.connect(self.compare_models_dialog)
        tools_menu.addAction(compare_action)
        
        tools_menu.addSeparator()
        
        viz_action = QAction("3D Trajectory Visualization", self)
        viz_action.triggered.connect(self.show_trajectory_visualization)
        tools_menu.addAction(viz_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self.open_documentation)
        help_menu.addAction(docs_action)
        
        tutorial_action = QAction("Interactive Tutorial", self)
        tutorial_action.triggered.connect(self.show_tutorial)
        help_menu.addAction(tutorial_action)
        
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
        header = QLabel("VoxelPath Control Center v3.2")
        header_font = QFont("Consolas", 12, QFont.Weight.Bold)
        header.setFont(header_font)
        header.setStyleSheet("color: #00ffff; padding: 10px; background-color: #1a1a2a;")
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
        
        # Tab 4: Analytics
        analytics_tab = self._create_analytics_tab()
        tab_widget.addTab(analytics_tab, "Analytics")
        
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
        self.btn_load_corpus = QPushButton("📂 Load Training Corpus")
        self.btn_load_corpus.setMinimumHeight(40)
        self.btn_load_corpus.clicked.connect(self.load_corpus)
        self.lbl_corpus = QLabel("No corpus loaded")
        self.lbl_corpus.setWordWrap(True)
        self.lbl_corpus.setStyleSheet("color: #888888; font-style: italic;")
        corpus_layout.addWidget(self.btn_load_corpus)
        corpus_layout.addWidget(self.lbl_corpus)
        corpus_group.setLayout(corpus_layout)
        layout.addWidget(corpus_group)  # Now layout is properly defined
        
        # Training Parameters
        param_group = QGroupBox("Training Parameters")
        param_layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1000, 100000)
        self.spin_batch_size.setValue(10000)
        self.spin_batch_size.setSuffix(" tokens")
        batch_layout.addWidget(self.spin_batch_size)
        param_layout.addLayout(batch_layout)
        
        # Memory limit
        mem_layout = QHBoxLayout()
        mem_layout.addWidget(QLabel("Memory Limit:"))
        self.spin_memory_limit = QSpinBox()
        self.spin_memory_limit.setRange(100, 10000)
        self.spin_memory_limit.setValue(2000)
        self.spin_memory_limit.setSuffix(" MB")
        mem_layout.addWidget(self.spin_memory_limit)
        param_layout.addLayout(mem_layout)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Training Controls
        control_group = QGroupBox("Training Controls")
        control_layout = QVBoxLayout()
        self.btn_start_training = QPushButton("▶ Start Training")
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
        self.btn_stop_training = QPushButton("⏹ Stop Training")
        self.btn_stop_training.setMinimumHeight(35)
        self.btn_stop_training.setEnabled(False)
        self.btn_stop_training.clicked.connect(self.stop_training)
        self.btn_reset_engine = QPushButton("🔄 Reset Engine")
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

        # Training metrics table
        metrics_group = QGroupBox("Recent Training Metrics")
        metrics_layout = QVBoxLayout()
        self.table_training_metrics = QTableWidget(0, 11)
        self.table_training_metrics.setHorizontalHeaderLabels([
            'Batch', 'Tokens', '2-grams', '3-grams', '4-grams', '5-grams', 
            'Perplexity', 'Coherence', 'Diversity', 'Memory (MB)', 'Rate (tok/s)'
        ])
        self.table_training_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_training_metrics.verticalHeader().setVisible(False)
        self.table_training_metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_training_metrics.setFont(QFont("Consolas", 9))
        self.table_training_metrics.setAlternatingRowColors(True)
        metrics_layout.addWidget(self.table_training_metrics)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Spacer
        layout.addStretch()
        
        return tab

    def compare_models_dialog(self):
        """Open dialog to compare multiple selected models"""
        selected_items = self.list_models.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Selection Error", 
                            "Please select at least 2 models to compare (hold Ctrl to select multiple).")
            return
            
        model_paths = []
        model_names = []
        for item in selected_items:
            model_path = item.data(Qt.ItemDataRole.UserRole)
            model_paths.append(model_path)
            model_names.append(os.path.basename(model_path))
            
        try:
            # Get model comparison data
            comparison = self.engine.compare_models(model_paths)
            
            # Create comparison dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Model Comparison")
            dialog.setMinimumSize(900, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Create table
            table = QTableWidget(len(comparison), 8)
            table.setHorizontalHeaderLabels([
                "Model", "Tokens", "2-grams", "3-grams", "4-grams", "5-grams", 
                "Memory (MB)", "Perplexity"
            ])
            
            # Populate table
            row = 0
            for model_name, stats in comparison.items():
                if "error" in stats:
                    table.setItem(row, 0, QTableWidgetItem(model_name))
                    table.setItem(row, 1, QTableWidgetItem(f"Error: {stats['error']}"))
                    row += 1
                    continue
                    
                table.setItem(row, 0, QTableWidgetItem(model_name))
                table.setItem(row, 1, QTableWidgetItem(f"{stats.get('tokens_processed', 0):,}"))
                
                # N-gram counts
                ngrams = stats.get("ngrams_learned", {})
                table.setItem(row, 2, QTableWidgetItem(f"{ngrams.get(2, 0):,}"))
                table.setItem(row, 3, QTableWidgetItem(f"{ngrams.get(3, 0):,}"))
                table.setItem(row, 4, QTableWidgetItem(f"{ngrams.get(4, 0):,}"))
                table.setItem(row, 5, QTableWidgetItem(f"{ngrams.get(5, 0):,}"))
                
                table.setItem(row, 6, QTableWidgetItem(f"{stats.get('memory_usage_mb', 0):.1f}"))
                table.setItem(row, 7, QTableWidgetItem(f"{stats.get('perplexity', float('inf')):.2f}"))
                
                row += 1
                
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            layout.addWidget(table)
            
            # Add close button
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.accept)
            layout.addWidget(button_box)
            
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Comparison Error", f"Failed to compare models:\n{str(e)}")
            self._system_log(f"Comparison error: {str(e)}", "ERROR")
        
    def _create_generation_tab(self) -> QWidget:
        """Create text generation tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # Seed Input
        seed_group = QGroupBox("Seed Text")
        seed_layout = QVBoxLayout()
        self.txt_seed = QTextEdit()
        self.txt_seed.setPlaceholderText("Enter seed text (e.g., 'It was the best of times')")
        self.txt_seed.setMaximumHeight(100)
        self.txt_seed.setFont(QFont("Consolas", 10))
        seed_layout.addWidget(self.txt_seed)
        seed_group.setLayout(seed_layout)
        layout.addWidget(seed_group)
        
        # Generation Parameters
        param_group = QGroupBox("Generation Parameters")
        param_layout = QGridLayout()
        
        # Temperature
        param_layout.addWidget(QLabel("Temperature:"), 0, 0)
        self.slider_temp = QSlider(Qt.Orientation.Horizontal)
        self.slider_temp.setRange(10, 200)
        self.slider_temp.setValue(85)
        self.slider_temp.valueChanged.connect(self._update_temp_label)
        param_layout.addWidget(self.slider_temp, 0, 1)
        self.lbl_temp_value = QLabel("0.85")
        self.lbl_temp_value.setMinimumWidth(40)
        param_layout.addWidget(self.lbl_temp_value, 0, 2)
        
        # Top-P
        param_layout.addWidget(QLabel("Top-P:"), 1, 0)
        self.slider_top_p = QSlider(Qt.Orientation.Horizontal)
        self.slider_top_p.setRange(10, 100)
        self.slider_top_p.setValue(90)
        self.slider_top_p.valueChanged.connect(self._update_top_p_label)
        param_layout.addWidget(self.slider_top_p, 1, 1)
        self.lbl_top_p_value = QLabel("0.90")
        self.lbl_top_p_value.setMinimumWidth(40)
        param_layout.addWidget(self.lbl_top_p_value, 1, 2)
        
        # Repetition Penalty
        param_layout.addWidget(QLabel("Repetition Penalty:"), 2, 0)
        self.slider_rep_penalty = QSlider(Qt.Orientation.Horizontal)
        self.slider_rep_penalty.setRange(5, 30)
        self.slider_rep_penalty.setValue(12)
        self.slider_rep_penalty.valueChanged.connect(self._update_rep_penalty_label)
        param_layout.addWidget(self.slider_rep_penalty, 2, 1)
        self.lbl_rep_penalty_value = QLabel("1.2")
        self.lbl_rep_penalty_value.setMinimumWidth(40)
        param_layout.addWidget(self.lbl_rep_penalty_value, 2, 2)
        
        # Length
        param_layout.addWidget(QLabel("Output Length:"), 3, 0)
        self.spin_output_length = QSpinBox()
        self.spin_output_length.setRange(50, 5000)
        self.spin_output_length.setValue(300)
        self.spin_output_length.setSuffix(" tokens")
        param_layout.addWidget(self.spin_output_length, 3, 1, 1, 2)
        
        # Minimum length
        param_layout.addWidget(QLabel("Minimum Length:"), 4, 0)
        self.spin_min_length = QSpinBox()
        self.spin_min_length.setRange(10, 1000)
        self.spin_min_length.setValue(100)
        self.spin_min_length.setSuffix(" tokens")
        param_layout.addWidget(self.spin_min_length, 4, 1, 1, 2)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Advanced Features
        advanced_group = QGroupBox("Advanced Features")
        advanced_layout = QGridLayout()
        
        self.cb_use_astar = QCheckBox("Enable A* Pathfinding")
        self.cb_use_astar.setToolTip("Use A* search for coherent trajectory through voxel space")
        advanced_layout.addWidget(self.cb_use_astar, 0, 0)
        
        self.cb_use_momentum = QCheckBox("Enable Geometric Momentum")
        self.cb_use_momentum.setChecked(True)
        self.cb_use_momentum.setToolTip("Maintain velocity and acceleration through token space")
        advanced_layout.addWidget(self.cb_use_momentum, 0, 1)
        
        self.cb_use_tunnels = QCheckBox("Enable Trajectory Tunnels")
        self.cb_use_tunnels.setChecked(True)
        self.cb_use_tunnels.setToolTip("Use learned shortcuts for common patterns")
        advanced_layout.addWidget(self.cb_use_tunnels, 1, 0)
        
        self.cb_adaptive_temp = QCheckBox("Adaptive Temperature")
        self.cb_adaptive_temp.setToolTip("Automatically adjust temperature based on context")
        advanced_layout.addWidget(self.cb_adaptive_temp, 1, 1)
        
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
        self.btn_analyze = QPushButton("Analyze Generated Text")
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
        self.txt_output.setFont(QFont("Georgia", 11))
        self.txt_output.setStyleSheet("""
        QTextEdit {
            background-color: #1a1a2e;
            color: #e0e0e0;
            border: 1px solid #333344;
            font-family: 'Georgia';
            line-height: 1.4;
        }
        """)
        self.txt_output.setMaximumHeight(200)
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
        model_control_layout = QGridLayout()
        
        self.btn_refresh_models = QPushButton("Refresh List")
        self.btn_refresh_models.clicked.connect(self.refresh_model_list)
        model_control_layout.addWidget(self.btn_refresh_models, 0, 0)
        
        self.btn_load_model = QPushButton("Load Selected Model")
        self.btn_load_model.clicked.connect(self.load_selected_model)
        model_control_layout.addWidget(self.btn_load_model, 0, 1)
        
        self.btn_delete_model = QPushButton("Delete Selected Model")
        self.btn_delete_model.setStyleSheet("background-color: #5d4037; color: white;")
        self.btn_delete_model.clicked.connect(self.delete_selected_model)
        model_control_layout.addWidget(self.btn_delete_model, 1, 0)
        
        self.btn_export_model = QPushButton("Export Model")
        self.btn_export_model.clicked.connect(self.export_model)
        model_control_layout.addWidget(self.btn_export_model, 1, 1)
        
        self.btn_compare_models = QPushButton("Compare Selected Models")
        self.btn_compare_models.clicked.connect(self.compare_selected_models)
        model_control_layout.addWidget(self.btn_compare_models, 2, 0, 1, 2)
        
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
        
    def _create_analytics_tab(self) -> QWidget:
        """Create analytics and visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # Training Metrics Visualization
        viz_group = QGroupBox("Training Metrics Visualization")
        viz_layout = QVBoxLayout()
        
        self.training_viz = TrainingProgressVisualizer()
        viz_layout.addWidget(self.training_viz)
        
        # Controls
        control_layout = QHBoxLayout()
        self.btn_refresh_viz = QPushButton("Refresh Visualization")
        self.btn_refresh_viz.clicked.connect(self.refresh_training_visualization)
        self.btn_clear_viz = QPushButton("Clear Visualization")
        self.btn_clear_viz.clicked.connect(self.clear_training_visualization)
        self.btn_export_viz = QPushButton("Export Metrics")
        self.btn_export_viz.clicked.connect(self.export_training_metrics)
        
        control_layout.addWidget(self.btn_refresh_viz)
        control_layout.addWidget(self.btn_clear_viz)
        control_layout.addWidget(self.btn_export_viz)
        control_layout.addStretch()
        
        viz_layout.addLayout(control_layout)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Quality Metrics
        quality_group = QGroupBox("Text Quality Benchmarks")
        quality_layout = QVBoxLayout()
        
        self.table_quality_metrics = QTableWidget(0, 3)
        self.table_quality_metrics.setHorizontalHeaderLabels(["Metric", "Value", "Benchmark"])
        self.table_quality_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_quality_metrics.verticalHeader().setVisible(False)
        self.table_quality_metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        quality_layout.addWidget(self.table_quality_metrics)
        
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
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
        viz_header.setStyleSheet("color: #00ffff; padding: 5px; background-color: #1a1a2a;")
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
        self.combo_colormap.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'cool', 'jet', 'hsv'])
        self.combo_colormap.setCurrentText('viridis')
        self.combo_colormap.currentTextChanged.connect(self.refresh_visualization)
        viz_controls.addWidget(self.combo_colormap)
        
        # Point size
        viz_controls.addWidget(QLabel("Point Size:"))
        self.spin_point_size = QSpinBox()
        self.spin_point_size.setRange(1, 10)
        self.spin_point_size.setValue(1)
        self.spin_point_size.valueChanged.connect(self.refresh_visualization)
        viz_controls.addWidget(self.spin_point_size)
        
        # Refresh and clear buttons
        self.btn_refresh_viz = QPushButton("Refresh")
        self.btn_refresh_viz.clicked.connect(self.refresh_visualization)
        viz_controls.addWidget(self.btn_refresh_viz)
        
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
        self.ax.set_xlabel('X (Char 1)', color='white', fontsize=10, labelpad=8)
        self.ax.set_ylabel('Y (Char 2)', color='white', fontsize=10, labelpad=8)
        self.ax.set_zlabel('Z (Char 3)', color='white', fontsize=10, labelpad=8)
        
        # Set title
        self.ax.set_title('3D Voxel Token Space', color='white', fontsize=12, pad=10)
        
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
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
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
        elif level == "DEBUG":
            color = "#aaaaff"
            prefix = "DEBUG"
        else:
            color = "#00aaff"
            prefix = "INFO"
            
        # Format message
        formatted = f'<span style="color: #888888;">[{timestamp}]</span> ' \
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
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            seconds = uptime % 60
            
            status = f"Tokens: {self.engine.tokens_processed:,} | " \
                    f"N-Grams: {sum(self.engine.ngrams_learned.values()):,} | " \
                    f"Memory: {self.engine.memory_usage_mb:.1f} MB | " \
                    f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}"
                    
            self.status_bar.showMessage(status)
            
    def update_training_statistics(self, metrics: dict):
        """Update training statistics display"""
        if not metrics:
            return
            
        # Update table
        row = self.table_training_metrics.rowCount()
        self.table_training_metrics.insertRow(row)
        
        # Add data to row
        items = [
            QTableWidgetItem(str(metrics.get('batch_idx', row) + 1)),
            QTableWidgetItem(f"{metrics.get('tokens_processed', 0):,}"),
            QTableWidgetItem(f"{metrics.get('ngrams_learned', {}).get(2, 0):,}"),
            QTableWidgetItem(f"{metrics.get('ngrams_learned', {}).get(3, 0):,}"),
            QTableWidgetItem(f"{metrics.get('ngrams_learned', {}).get(4, 0):,}"),
            QTableWidgetItem(f"{metrics.get('ngrams_learned', {}).get(5, 0):,}"),
            QTableWidgetItem(f"{metrics.get('perplexity', float('inf')):.2f}"),
            QTableWidgetItem(f"{metrics.get('coherence_score', 0.0):.2f}"),
            QTableWidgetItem(f"{metrics.get('diversity_score', 0.0):.2f}"),
            QTableWidgetItem(f"{metrics.get('memory_usage_mb', 0.0):.1f}"),
            QTableWidgetItem(f"{metrics.get('training_rate', 0.0):.0f}")
        ]
        
        for col, item in enumerate(items):
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table_training_metrics.setItem(row, col, item)
            
        # Keep only the last 20 rows for performance
        if self.table_training_metrics.rowCount() > 20:
            self.table_training_metrics.removeRow(0)
            
        # Update metrics history for visualization
        self.training_metrics_history.append(metrics)
        self.training_viz.update_metrics(self.training_metrics_history[-50:])  # Keep last 50
        
        # Update quality benchmarks
        self.update_quality_benchmarks(metrics)
        
    def update_quality_benchmarks(self, metrics: dict):
        """Update quality benchmark table"""
        self.table_quality_metrics.setRowCount(0)
        
        benchmark_data = [
            ("Perplexity", metrics.get('perplexity', float('inf')), "Lower is better (<150 good)"),
            ("Coherence Score", metrics.get('coherence_score', 0.0), "Higher is better (>0.6 good)"),
            ("Diversity Score", metrics.get('diversity_score', 0.0), "Higher is better (>0.4 good)"),
            ("Memory Usage", self.engine.memory_usage_mb, "Lower is better (<2000 MB good)"),
            ("Training Rate", metrics.get('training_rate', 0.0), "Higher is better (>1000 tok/s good)"),
            ("2-grams Learned", self.engine.ngrams_learned.get(2, 0), "More is better"),
            ("5-grams Learned", self.engine.ngrams_learned.get(5, 0), "More is better"),
        ]
        
        for i, (metric, value, benchmark) in enumerate(benchmark_data):
            self.table_quality_metrics.insertRow(i)
            
            # Metric name
            item_metric = QTableWidgetItem(metric)
            self.table_quality_metrics.setItem(i, 0, item_metric)
            
            # Value with conditional formatting
            item_value = QTableWidgetItem(f"{value:.2f}" if isinstance(value, float) else f"{value:,}")
            if metric == "Perplexity":
                if value < 150:
                    item_value.setBackground(QColor(50, 150, 50))
                    item_value.setForeground(QColor(255, 255, 255))
                elif value < 300:
                    item_value.setBackground(QColor(150, 150, 50))
                    item_value.setForeground(QColor(0, 0, 0))
                else:
                    item_value.setBackground(QColor(150, 50, 50))
                    item_value.setForeground(QColor(255, 255, 255))
            elif metric in ["Coherence Score", "Diversity Score"]:
                if value > 0.6:
                    item_value.setBackground(QColor(50, 150, 50))
                    item_value.setForeground(QColor(255, 255, 255))
                elif value > 0.4:
                    item_value.setBackground(QColor(150, 150, 50))
                    item_value.setForeground(QColor(0, 0, 0))
                else:
                    item_value.setBackground(QColor(150, 50, 50))
                    item_value.setForeground(QColor(255, 255, 255))
                    
            self.table_quality_metrics.setItem(i, 1, item_value)
            
            # Benchmark description
            item_bench = QTableWidgetItem(benchmark)
            self.table_quality_metrics.setItem(i, 2, item_bench)
            
    def refresh_training_visualization(self):
        """Refresh the training metrics visualization"""
        self.training_viz.update_metrics(self.training_metrics_history[-100:])  # Show last 100 points
        
    def clear_training_visualization(self):
        """Clear the training visualization"""
        self.training_viz.clear()
        self._system_log("Training visualization cleared")
        
    def export_training_metrics(self):
        """Export training metrics to CSV file"""
        if not self.training_metrics_history:
            QMessageBox.warning(self, "No Data", "No training metrics to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Training Metrics",
            f"training_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    headers = [
                        'timestamp', 'batch_idx', 'tokens_processed', '2_grams', '3_grams', 
                        '4_grams', '5_grams', 'perplexity', 'coherence_score', 
                        'diversity_score', 'memory_usage_mb', 'training_rate', 'batch_time'
                    ]
                    writer.writerow(headers)
                    
                    # Data
                    for metrics in self.training_metrics_history:
                        row = [
                            metrics.get('timestamp', datetime.datetime.now().isoformat()),
                            metrics.get('batch_idx', 0),
                            metrics.get('tokens_processed', 0),
                            metrics.get('ngrams_learned', {}).get(2, 0),
                            metrics.get('ngrams_learned', {}).get(3, 0),
                            metrics.get('ngrams_learned', {}).get(4, 0),
                            metrics.get('ngrams_learned', {}).get(5, 0),
                            metrics.get('perplexity', float('inf')),
                            metrics.get('coherence_score', 0.0),
                            metrics.get('diversity_score', 0.0),
                            metrics.get('memory_usage_mb', 0.0),
                            metrics.get('training_rate', 0.0),
                            metrics.get('batch_time', 0.0)
                        ]
                        writer.writerow(row)
                        
                self._system_log(f"Training metrics exported to: {file_path}", "SUCCESS")
                QMessageBox.information(self, "Export Complete", 
                                      f"Training metrics saved to:\n{file_path}")
            except Exception as e:
                self._system_log(f"Export error: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Export Error", f"Failed to export metrics:\n{str(e)}")
                
    # ============================================================================
    # CORE FUNCTIONALITY
    # ============================================================================
    def load_corpus(self):
        """Load training corpus with size validation and preview"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Corpus",
            "",
            "Text Files (*.txt *.csv *.json);;All Files (*.*)"
        )
        if file_path:
            try:
                # Get file size first
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                # Show preview and get confirmation for large files
                if file_size > 100:
                    reply = QMessageBox.warning(
                        self,
                        "Large File Warning",
                        f"This file is {file_size:.1f} MB. Loading very large files may impact performance and memory usage.\n"
                        "Only the first 10MB will be used for initial training. Continue?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                        
                # Read file with size limitation
                max_chars = 10_000_000  # 10MB limit
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    if file_size > 10:
                        # Read only first 10MB
                        corpus_text = f.read(max_chars)
                        self._system_log(f"Loaded first {max_chars:,} characters from large file", "WARNING")
                    else:
                        corpus_text = f.read()
                        
                # Update UI
                self.current_corpus = file_path
                self.lbl_corpus.setText(f"{os.path.basename(file_path)} ({file_size:.1f} MB)")
                
                # Show preview
                preview = corpus_text[:500] + "..." if len(corpus_text) > 500 else corpus_text
                QMessageBox.information(self, "Corpus Loaded", 
                                      f"File: {os.path.basename(file_path)}\n"
                                      f"Size: {file_size:.1f} MB\n"
                                      f"Characters: {len(corpus_text):,}\n\n"
                                      f"Preview:\n{preview}")
                
                self._system_log(f"Corpus loaded: {os.path.basename(file_path)} ({len(corpus_text):,} chars)")
            except Exception as e:
                self._system_log(f"Error loading corpus: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Load Error", f"Failed to load corpus:\n{str(e)}")
                
    def load_validation_corpus(self):
        """Load validation corpus"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Validation Corpus",
            "",
            "Text Files (*.txt *.csv *.json);;All Files (*.*)"
        )
        if file_path:
            try:
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                # Read file with size limitation (smaller for validation)
                max_chars = 1_000_000  # 1MB limit for validation
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    if file_size > 1:
                        validation_text = f.read(max_chars)
                        self._system_log(f"Loaded first {max_chars:,} characters from validation file", "WARNING")
                    else:
                        validation_text = f.read()
                        
                self.validation_corpus = file_path
                self.lbl_val_corpus.setText(f"{os.path.basename(file_path)} ({file_size:.1f} MB)")
                
                self._system_log(f"Validation corpus loaded: {os.path.basename(file_path)}")
            except Exception as e:
                self._system_log(f"Error loading validation corpus: {str(e)}", "ERROR")
                
    def start_training(self):
        """Start training process with validation support"""
        if not hasattr(self, 'current_corpus') or not self.current_corpus:
            QMessageBox.warning(self, "No Corpus", "Please load a training corpus first.")
            return
            
        # Reset training history
        self.training_metrics_history = []
        self.validation_results = []
        self.table_training_metrics.setRowCount(0)
        
        # Disable controls during training
        self.btn_start_training.setEnabled(False)
        self.btn_stop_training.setEnabled(True)
        self.btn_load_corpus.setEnabled(False)
        self.btn_load_val.setEnabled(False)
        
        # Reset progress
        self.progress_training.setValue(0)
        self.lbl_training_status.setText("Training in progress...")
        self.lbl_training_stats.setText("Initializing training...")

        if hasattr(self, 'btn_load_val'):
            self.btn_load_val.setEnabled(False)
        
        # Get validation path if available
        validation_path = getattr(self, 'validation_corpus', None)
        
        # Create and start worker
        self.training_worker = TrainingWorker(
            self.engine, 
            self.current_corpus, 
            validation_path
        )
        self.training_worker.progress_updated.connect(self.on_training_progress)
        self.training_worker.training_complete.connect(self.on_training_complete)
        self.training_worker.error_occurred.connect(self.on_training_error)
        self.training_worker.validation_metrics.connect(self.on_validation_metrics)
        self.training_worker.start()
        self._system_log("Training started with validation support", "INFO")
        
    def on_training_progress(self, progress: int, message: str, metrics: dict):
        """Handle training progress updates"""
        self.progress_training.setValue(progress)
        self.lbl_training_status.setText(message)
        
        if metrics:
            self.lbl_training_stats.setText(
                f"Tokens: {metrics.get('tokens_processed', 0):,} | "
                f"N-Grams: {sum(metrics.get('ngrams_learned', {}).values()):,} | "
                f"Perplexity: {metrics.get('perplexity', float('inf')):.2f}"
            )
            self.update_training_statistics(metrics)
            
    def on_validation_metrics(self, metrics: dict):
        """Handle validation metrics updates"""
        if metrics:
            self.validation_results.append(metrics)
            perplexity = metrics.get('perplexity', float('inf'))
            coherence = metrics.get('coherence_score', 0.0)
            diversity = metrics.get('diversity_score', 0.0)
            
            self._system_log(f"Validation: Perplexity={perplexity:.2f}, "
                           f"Coherence={coherence:.2f}, Diversity={diversity:.2f}", 
                           "SUCCESS")
                           
    def on_training_complete(self, message: str, metrics: dict):
        """Handle training completion"""
        # Re-enable controls
        self.btn_start_training.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        self.btn_load_corpus.setEnabled(True)
        self.btn_load_val.setEnabled(True)
        
        self.lbl_training_status.setText("Training complete")
        self.progress_training.setValue(100)
        
        if metrics:
            final_msg = (
                f"Training completed successfully!\n\n"
                f"Total Tokens Processed: {metrics['tokens_processed']:,}\n"
                f"Total N-Grams Learned: {metrics['total_ngrams']:,}\n"
                f"Final Average Perplexity: {metrics['avg_perplexity']:.2f}\n"
                f"Training Time: {metrics['training_time']:.1f} seconds\n"
                f"Memory Usage: {metrics['memory_usage_mb']:.1f} MB"
            )
            self._system_log(final_msg, "SUCCESS")
            QMessageBox.information(self, "Training Complete", final_msg)
            
        # Update UI
        self.refresh_visualization()
        self.refresh_model_list()
        
    def on_training_error(self, error_message: str):
        """Handle training errors"""
        self.btn_start_training.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        self.btn_load_corpus.setEnabled(True)
        self.btn_load_val.setEnabled(True)
        
        self.lbl_training_status.setText("Training failed")
        self.progress_training.setValue(0)
        
        self._system_log(error_message, "ERROR")
        QMessageBox.critical(self, "Training Error", error_message)
        
    def stop_training(self):
        """Stop training process gracefully"""
        if self.training_worker and self.training_worker.isRunning():
            self._system_log("Stopping training process...", "WARNING")
            self.lbl_training_status.setText("Stopping training...")
            
            self.training_worker.stop()
            # Wait with timeout
            if not self.training_worker.wait(5000):  # 5 seconds timeout
                self._system_log("Training stop timed out, forcing termination", "ERROR")
                
            self.btn_start_training.setEnabled(True)
            self.btn_stop_training.setEnabled(False)
            self.lbl_training_status.setText("Training stopped by user")
            self._system_log("Training stopped by user", "WARNING")
            
    def reset_engine(self):
        """Reset the engine to initial state with confirmation"""
        reply = QMessageBox.question(
            self,
            "Reset Engine",
            "This will delete all learned patterns and reset the engine to initial state.\n"
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.engine.reset_engine()
                self.training_metrics_history = []
                self.validation_results = []
                self.table_training_metrics.setRowCount(0)
                self.training_viz.clear()
                
                # Clear outputs
                self.txt_output.clear()
                self.txt_full_output.clear()
                self.current_trajectory = []
                
                # Update UI
                self.refresh_visualization()
                self.refresh_model_list()
                
                self._system_log("Engine reset complete", "SUCCESS")
                QMessageBox.information(self, "Reset Complete", 
                                      "Engine has been reset to initial state.")
            except Exception as e:
                self._system_log(f"Reset error: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Reset Error", f"Failed to reset engine:\n{str(e)}")
                
    def generate_text(self):
        """Generate text based on current parameters"""
        seed_text = self.txt_seed.toPlainText().strip()
        if not seed_text:
            # Use a default seed based on time
            hour = datetime.datetime.now().hour
            if 5 <= hour < 12:
                default_seed = "the morning sun rose over the horizon"
            elif 12 <= hour < 17:
                default_seed = "the afternoon sun cast long shadows"
            elif 17 <= hour < 21:
                default_seed = "the evening stars began to appear"
            else:
                default_seed = "the night was dark and full of secrets"
                
            reply = QMessageBox.question(
                self,
                "No Seed Text",
                f"No seed text entered. Would you like to use this default seed:\n\n'{default_seed}'\n\nOr cancel to enter your own?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                seed_text = default_seed
                self.txt_seed.setPlainText(seed_text)
            elif reply == QMessageBox.StandardButton.Cancel:
                return
                
        # Get parameters
        temperature = self.slider_temp.value() / 100.0
        top_p = self.slider_top_p.value() / 100.0
        repetition_penalty = self.slider_rep_penalty.value() / 10.0
        max_length = self.spin_output_length.value()
        min_length = self.spin_min_length.value()
        use_astar = self.cb_use_astar.isChecked()
        use_momentum = self.cb_use_momentum.isChecked()
        
        # Advanced features
        use_tunnels = self.cb_use_tunnels.isChecked()
        adaptive_temp = self.cb_adaptive_temp.isChecked()
        
        # Disable button during generation
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("Generating...")
        self.lbl_training_status.setText("Generating text...")
        
        # Create and start generation worker
        self.generation_worker = GenerationWorker(
            self.engine,
            seed_text,
            max_length,
            temperature,
            top_p,
            repetition_penalty,
            use_astar,
            use_momentum,
            min_length
        )
        self.generation_worker.generation_progress.connect(self.on_generation_progress)
        self.generation_worker.generation_complete.connect(self.on_generation_complete)
        self.generation_worker.start()
        
        self._system_log(f"Generating text with seed: '{seed_text[:50]}...'", "INFO")
        
    def on_generation_progress(self, progress: int, message: str):
        """Handle generation progress updates"""
        self.btn_generate.setText(f"Generating... {progress}%")
        self.lbl_training_status.setText(message)
        
    def on_generation_complete(self, generated_text: str, trajectory: list, metrics: dict):
        """Handle generation completion with full metrics analysis"""
        # Re-enable button
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("Generate Text")
        self.lbl_training_status.setText("Ready")
        
        if generated_text.startswith("Error:"):
            self._system_log(generated_text, "ERROR")
            QMessageBox.critical(self, "Generation Error", generated_text)
            return
            
        # Update output displays
        preview_length = min(500, len(generated_text))
        self.txt_output.setText(generated_text[:preview_length] + 
                              ("..." if len(generated_text) > preview_length else ""))
        self.txt_full_output.setText(generated_text)
        self.last_generated_text = generated_text
        self.last_generation_metrics = metrics
        
        # Update trajectory and visualization
        self.current_trajectory = trajectory
        self.refresh_visualization()
        
        # Update status with key metrics
        coherence = metrics.get('coherence_score', 0.0)
        diversity = metrics.get('diversity_score', 0.0)
        rep_rate = metrics.get('repetition_rate', 0.0) * 100
        char_count = metrics.get('character_count', len(generated_text))
        
        status_msg = (f"Generated {char_count:,} characters | "
                     f"Coherence: {coherence:.2f} | "
                     f"Diversity: {diversity:.2f} | "
                     f"Repetition: {rep_rate:.1f}%")
        self.status_bar.showMessage(status_msg, 8000)
        
        # Log completion with metrics summary
        log_msg = (f"Generation complete: {char_count:,} chars | "
                  f"Coherence: {coherence:.2f} | "
                  f"Diversity: {diversity:.2f} | "
                  f"Time: {metrics.get('generation_time', 0.0):.2f}s")
        self._system_log(log_msg, "SUCCESS")
        
        # Auto-analyze if metrics indicate issues
        if coherence < 0.4 or diversity < 0.3 or rep_rate > 30.0:
            self._system_log("Quality metrics indicate potential issues. Consider analyzing the text.", "WARNING")
            # Auto-show analysis dialog after 2 seconds
            QTimer.singleShot(2000, lambda: self.analyze_text(auto_show=True))
            
    def analyze_text(self, auto_show=False):
        """Analyze the generated text with comprehensive metrics"""
        if not hasattr(self, 'last_generated_text') or not self.last_generated_text:
            QMessageBox.warning(self, "No Text", "Please generate text first.")
            return
            
        if not hasattr(self, 'last_generation_metrics') or not self.last_generation_metrics:
            # Fallback to engine analysis
            tokens = self.engine.tokenizer.tokenize(self.last_generated_text)
            analysis = self.engine.tokenizer.analyze_text(tokens)
            metrics = {
                'perplexity': analysis.perplexity,
                'coherence_score': analysis.coherence_score,
                'diversity_score': analysis.diversity_score,
                'repetition_rate': analysis.repetition_rate,
                'avg_word_length': analysis.avg_word_length,
                'sentence_complexity': analysis.sentence_complexity,
                'grammatical_correctness': analysis.grammatical_correctness,
                'topic_consistency': analysis.topic_consistency,
                'character_count': len(self.last_generated_text),
                'token_count': len(tokens)
            }
        else:
            metrics = self.last_generation_metrics
            
        # Show analysis dialog
        dialog = TextAnalysisDialog(metrics, self.last_generated_text[:1000], self)
        
        if auto_show:
            dialog.show()
        else:
            dialog.exec()
            
    def clear_output(self):
        """Clear all output displays"""
        self.txt_output.clear()
        self.txt_full_output.clear()
        self.current_trajectory = []
        self.last_generated_text = ""
        self.last_generation_metrics = {}
        self.refresh_visualization()
        self._system_log("Output cleared")
        
    def refresh_visualization(self):
        """Refresh the 3D visualization with improved rendering"""
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
            self.ax.set_xlabel('X (Char 1)', color='white', fontsize=10, labelpad=8)
            self.ax.set_ylabel('Y (Char 2)', color='white', fontsize=10, labelpad=8)
            self.ax.set_zlabel('Z (Char 3)', color='white', fontsize=10, labelpad=8)
            self.ax.set_title('3D Voxel Token Space', color='white', fontsize=12, pad=10)
            
            # Get density threshold from slider
            density_percent = self.slider_density.value()
            max_density = np.max(self.engine.voxel_tensor) if np.any(self.engine.voxel_tensor) else 1.0
            threshold = max_density * (1.0 - density_percent / 100.0)
            
            # Create voxel cloud with performance optimization
            indices = np.where(self.engine.voxel_tensor > threshold)
            if len(indices[0]) > 0:
                # Limit points for performance
                max_points = 10000
                if len(indices[0]) > max_points:
                    # Weighted random sampling based on density
                    densities = self.engine.voxel_tensor[indices]
                    probabilities = densities / np.sum(densities)
                    sample_idx = np.random.choice(
                        len(indices[0]),
                        min(max_points, len(indices[0])),
                        replace=False,
                        p=probabilities
                    )
                    x = indices[0][sample_idx]
                    y = indices[1][sample_idx]
                    z = indices[2][sample_idx]
                    c = densities[sample_idx]
                else:
                    x, y, z = indices
                    c = self.engine.voxel_tensor[x, y, z]
                    
                # Get selected colormap
                cmap_name = self.combo_colormap.currentText()
                
                # Create scatter plot with color based on density
                scatter = self.ax.scatter(
                    x, y, z,
                    c=c,
                    cmap=cmap_name,
                    alpha=0.3,
                    s=self.spin_point_size.value(),
                    label="Voxel Density"
                )
                
                # Add colorbar
                cbar = self.fig.colorbar(scatter, ax=self.ax, pad=0.1)
                cbar.set_label('Density', color='white', fontsize=9)
                cbar.ax.tick_params(colors='white')
                cbar.outline.set_edgecolor('white')
                
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
                
                # Plot start and end points with annotations
                start_pos = trajectory[0]
                end_pos = trajectory[-1]
                
                self.ax.scatter(
                    start_pos[0], start_pos[1], start_pos[2],
                    color='#00ff00',
                    s=150,
                    edgecolors='white',
                    linewidth=1.5,
                    zorder=10
                )
                self.ax.text(
                    start_pos[0], start_pos[1], start_pos[2],
                    'Start', 
                    color='#00ff00',
                    fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none')
                )
                
                self.ax.scatter(
                    end_pos[0], end_pos[1], end_pos[2],
                    color='#ff0000',
                    s=150,
                    edgecolors='white',
                    linewidth=1.5,
                    zorder=10
                )
                self.ax.text(
                    end_pos[0], end_pos[1], end_pos[2],
                    'End', 
                    color='#ff0000',
                    fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none')
                )
                
                # Annotate key points along trajectory
                if len(trajectory) > 10:
                    step = max(1, len(trajectory) // 5)
                    for i in range(0, len(trajectory), step):
                        pos = trajectory[i]
                        self.ax.text(
                            pos[0], pos[1], pos[2],
                            f'Step {i}', 
                            color='#00aaff',
                            fontsize=8,
                            alpha=0.8
                        )
                        
            # Set optimal viewing angle based on trajectory
            if self.current_trajectory:
                trajectory = np.array(self.current_trajectory)
                centroid = np.mean(trajectory, axis=0)
                self.ax.view_init(elev=20, azim=45)
            else:
                self.ax.view_init(elev=25, azim=35)
                
            # Set equal aspect ratio if possible
            try:
                self.ax.set_box_aspect([1,1,1])
            except:
                pass
                
            # Create legend only if there are artists with labels
            if self.ax.get_legend_handles_labels()[0]:
                legend = self.ax.legend(
                    loc='upper right', 
                    fontsize='small', 
                    facecolor='#1a1a1a',
                    edgecolor='#333333'
                )
                for text in legend.get_texts():
                    text.set_color('white')
                    
            # Adjust layout and redraw
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            error_msg = f"Visualization error: {str(e)}"
            self._system_log(error_msg, "ERROR")
            QMessageBox.warning(self, "Visualization Error", error_msg)
            
    def clear_visualization(self):
        """Clear the visualization and trajectory"""
        self.current_trajectory = []
        self.refresh_visualization()
        self._system_log("Visualization cleared")
        
    def refresh_model_list(self):
        """Refresh the list of saved models with detailed information"""
        self.list_models.clear()
        self.loaded_models = []
        if not os.path.exists(self.engine.model_dir):
            os.makedirs(self.engine.model_dir, exist_ok=True)
            
        # Scan for model directories
        model_dirs = []
        for item in os.listdir(self.engine.model_dir):
            item_path = os.path.join(self.engine.model_dir, item)
            if os.path.isdir(item_path):
                # Check if it's a valid model
                metadata_path = os.path.join(item_path, "metadata.json")
                if os.path.exists(metadata_path):
                    model_dirs.append((item, item_path))
                    
        # Sort by modification time (newest first)
        model_dirs.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
        
        # Populate list
        for model_name, model_path in model_dirs:
            try:
                with open(os.path.join(model_path, "metadata.json"), 'r') as f:
                    metadata = json.load(f)
                    
                # Create list item with rich text
                item = QListWidgetItem()
                
                # Format display text
                token_count = metadata.get('tokens_processed', 0)
                ngrams_total = sum(metadata.get('ngrams_learned', {}).values())
                memory_mb = metadata.get('memory_usage_mb', 0)
                save_time = metadata.get('save_time', 'Unknown')
                
                # Parse ISO format time
                try:
                    save_dt = datetime.datetime.fromisoformat(save_time)
                    time_str = save_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    time_str = save_time[:16] if len(save_time) > 16 else save_time
                    
                display_text = (
                    f"<b>{model_name}</b><br>"
                    f"<span style='color: #88aa88;'>Tokens:</span> {token_count:,} | "
                    f"<span style='color: #8888aa;'>N-Grams:</span> {ngrams_total:,} | "
                    f"<span style='color: #aa8888;'>Memory:</span> {memory_mb:.1f} MB<br>"
                    f"<span style='color: #777777;'>Saved:</span> {time_str}"
                )
                
                item.setData(Qt.ItemDataRole.UserRole, model_path)
                item.setData(Qt.ItemDataRole.DisplayRole, display_text)
                item.setSizeHint(QSize(0, 60))  # Set item height
                
                self.list_models.addItem(item)
                self.loaded_models.append(model_path)
                
            except Exception as e:
                continue
                
        self._system_log(f"Found {self.list_models.count()} saved models")
        self.list_models.setStyleSheet("""
        QListWidget {
            background-color: #1a1a2a;
            border: 1px solid #333344;
            font-family: 'Consolas';
        }
        QListWidget::item {
            border-bottom: 1px solid #2a2a3a;
            padding: 5px;
        }
        QListWidget::item:selected {
            background-color: #2a3a5a;
            color: white;
        }
        """)
        
    def load_selected_model(self):
        """Load the selected model with progress feedback"""
        selected_items = self.list_models.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a model to load.")
            return
            
        model_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", "Model directory not found.")
            return
            
        # Show loading dialog
        loading_dialog = QDialog(self)
        loading_dialog.setWindowTitle("Loading Model")
        loading_dialog.setFixedSize(300, 100)
        loading_layout = QVBoxLayout(loading_dialog)
        
        progress_label = QLabel("Loading model components...")
        progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_layout.addWidget(progress_label)
        
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        loading_layout.addWidget(progress_bar)
        
        loading_dialog.show()
        QApplication.processEvents()
        
        try:
            # Load model with progress updates
            start_time = time.time()
            
            # Update progress
            progress_bar.setValue(20)
            progress_label.setText("Loading N-Gram memory...")
            QApplication.processEvents()
            
            # Load model
            success = self.engine.load_model(model_path)
            
            if success:
                # Update progress
                progress_bar.setValue(60)
                progress_label.setText("Updating visualizations...")
                QApplication.processEvents()
                
                # Refresh UI
                self.refresh_visualization()
                self.update_training_statistics({
                    'tokens_processed': self.engine.tokens_processed,
                    'ngrams_learned': self.engine.ngrams_learned,
                    'memory_usage_mb': self.engine.memory_usage_mb
                })
                self.refresh_model_list()
                
                # Update model info
                metadata_path = os.path.join(model_path, "metadata.json")
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                info_text = (
                    f"<b>Model:</b> {os.path.basename(model_path)}<br>"
                    f"<b>Tokens Processed:</b> {metadata['tokens_processed']:,}<br>"
                    f"<b>Total N-Grams:</b> {sum(metadata['ngrams_learned'].values()):,}<br>"
                    f"<b>Memory Usage:</b> {metadata.get('memory_usage_mb', 0):.1f} MB<br>"
                    f"<b>Training Time:</b> {metadata.get('training_time', 0):.1f} seconds<br>"
                    f"<b>Saved:</b> {metadata['save_time']}<br>"
                    f"<b>Version:</b> {metadata.get('model_version', 'unknown')}"
                )
                
                self.txt_model_info.setHtml(info_text)
                
                # Final progress update
                progress_bar.setValue(100)
                progress_label.setText("Model loaded successfully!")
                QApplication.processEvents()
                
                # Keep dialog visible for 0.5 seconds
                QTimer.singleShot(500, loading_dialog.accept)
                
                elapsed = time.time() - start_time
                self._system_log(f"Model loaded: {os.path.basename(model_path)} in {elapsed:.2f}s", "SUCCESS")
                QMessageBox.information(self, "Load Complete", 
                                      f"Model loaded successfully:\n{os.path.basename(model_path)}")
            else:
                loading_dialog.close()
                self._system_log("Failed to load model", "ERROR")
                
        except Exception as e:
            loading_dialog.close()
            error_msg = f"Failed to load model:\n{str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Load Error", error_msg)
            self._system_log(f"Load error: {str(e)}", "ERROR")
            
    def delete_selected_model(self):
        """Delete the selected model with confirmation"""
        selected_items = self.list_models.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a model to delete.")
            return
            
        model_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
        model_name = os.path.basename(model_path)
        
        reply = QMessageBox.question(
            self,
            "Delete Model",
            f"Are you sure you want to permanently delete this model?\n\n{model_name}\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Confirm deletion with additional safeguard
                confirm_text, ok = QInputDialog.getText(
                    self,
                    "Confirm Deletion",
                    f"Type the model name to confirm deletion:\n{model_name}",
                    QLineEdit.EchoMode.Normal,
                    ""
                )
                
                if ok and confirm_text.strip() == model_name:
                    # Remove directory and contents
                    import shutil
                    shutil.rmtree(model_path)
                    
                    # Refresh list
                    self.refresh_model_list()
                    
                    self._system_log(f"Model deleted: {model_name}", "SUCCESS")
                    QMessageBox.information(self, "Deletion Complete", 
                                          f"Model successfully deleted:\n{model_name}")
                else:
                    self._system_log("Model deletion canceled - name mismatch", "WARNING")
                    
            except Exception as e:
                error_msg = f"Failed to delete model:\n{str(e)}"
                QMessageBox.critical(self, "Delete Error", error_msg)
                self._system_log(f"Delete error: {str(e)}", "ERROR")
                
    def export_model(self):
        """Export current model to selected directory"""
        if not self.engine.tokens_processed:
            QMessageBox.warning(self, "No Model", "No trained model to export. Please train first.")
            return
            
        save_path = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            os.path.join(os.getcwd(), "exported_models")
        )
        
        if save_path:
            try:
                # Create export directory if needed
                export_dir = os.path.join(save_path, f"voxelpath_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(export_dir, exist_ok=True)
                
                # Show progress dialog
                progress_dialog = QDialog(self)
                progress_dialog.setWindowTitle("Exporting Model")
                progress_dialog.setFixedSize(400, 120)
                layout = QVBoxLayout(progress_dialog)
                
                status_label = QLabel("Starting export...")
                progress_bar = QProgressBar()
                progress_bar.setRange(0, 100)
                
                layout.addWidget(status_label)
                layout.addWidget(progress_bar)
                
                progress_dialog.show()
                QApplication.processEvents()
                
                # Export components with progress updates
                steps = [
                    ("Saving N-Gram memory", 30),
                    ("Saving voxel tensor", 20),
                    ("Saving spatial graph", 20),
                    ("Saving metadata", 30)
                ]
                
                total_progress = 0
                for step_name, step_weight in steps:
                    status_label.setText(step_name)
                    QApplication.processEvents()
                    
                    # Simulate progress within step
                    for i in range(1, 11):
                        progress_bar.setValue(total_progress + int(step_weight * i / 10))
                        QApplication.processEvents()
                        time.sleep(0.05)
                        
                    total_progress += step_weight
                    
                # Actually save the model
                export_path = self.engine.save_model(os.path.basename(export_dir))
                
                # Copy to export directory
                import shutil
                if os.path.exists(export_path):
                    shutil.copytree(export_path, export_dir, dirs_exist_ok=True)
                
                # Final progress
                progress_bar.setValue(100)
                status_label.setText("Export complete!")
                QApplication.processEvents()
                
                # Keep dialog visible briefly
                QTimer.singleShot(800, progress_dialog.accept)
                
                self._system_log(f"Model exported to: {export_dir}", "SUCCESS")
                QMessageBox.information(self, "Export Complete", 
                                      f"Model successfully exported to:\n{export_dir}")
                
            except Exception as e:
                error_msg = f"Failed to export model:\n{str(e)}"
                QMessageBox.critical(self, "Export Error", error_msg)
                self._system_log(f"Export error: {str(e)}", "ERROR")
                
    def compare_selected_models(self):
        """Compare multiple selected models with detailed metrics"""
        selected_items = self.list_models.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Selection Error", 
                              "Please select at least 2 models to compare (hold Ctrl to select multiple).")
            return
            
        model_paths = []
        model_names = []
        for item in selected_items:
            model_path = item.data(Qt.ItemDataRole.UserRole)
            model_paths.append(model_path)
            model_names.append(os.path.basename(model_path))
            
        try:
            # Show comparison dialog
            self._system_log(f"Comparing {len(model_paths)} models: {', '.join(model_names)}", "INFO")
            
            comparison = self.engine.compare_models(model_paths)
            
            # Create comparison dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Model Comparison")
            dialog.setMinimumSize(1000, 700)
            
            layout = QVBoxLayout()
            
            # Create tab widget for different comparison views
            tab_widget = QTabWidget()
            
            # Tab 1: Summary Table
            summary_tab = QWidget()
            summary_layout = QVBoxLayout(summary_tab)
            
            # Create table
            table = QTableWidget(len(comparison), 9)
            table.setHorizontalHeaderLabels([
                "Model", "Tokens", "2-grams", "3-grams", "4-grams", "5-grams", 
                "Memory (MB)", "Final Perplexity", "Best Perplexity"
            ])
            
            # Populate table
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
                table.setItem(row, 2, QTableWidgetItem(f"{ngrams.get('2', 0):,}"))
                table.setItem(row, 3, QTableWidgetItem(f"{ngrams.get('3', 0):,}"))
                table.setItem(row, 4, QTableWidgetItem(f"{ngrams.get('4', 0):,}"))
                table.setItem(row, 5, QTableWidgetItem(f"{ngrams.get('5', 0):,}"))
                
                table.setItem(row, 6, QTableWidgetItem(f"{stats.get('memory_usage', 0):.1f}"))
                table.setItem(row, 7, QTableWidgetItem(f"{stats.get('final_perplexity', float('inf')):.2f}"))
                table.setItem(row, 8, QTableWidgetItem(f"{stats.get('best_perplexity', float('inf')):.2f}"))
                
                row += 1
                
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.setFont(QFont("Consolas", 10))
            table.setAlternatingRowColors(True)
            
            # Apply conditional formatting for perplexity
            for row in range(table.rowCount()):
                final_perp_item = table.item(row, 7)
                best_perp_item = table.item(row, 8)
                
                if final_perp_item and best_perp_item:
                    try:
                        final_perp = float(final_perp_item.text())
                        best_perp = float(best_perp_item.text())
                        
                        # Color code based on perplexity values
                        def get_color(perp_value):
                            if perp_value < 100:
                                return QColor(50, 150, 50)  # Green
                            elif perp_value < 200:
                                return QColor(150, 150, 50)  # Yellow
                            else:
                                return QColor(150, 50, 50)  # Red
                                
                        final_perp_item.setBackground(get_color(final_perp))
                        final_perp_item.setForeground(QColor(255, 255, 255) if final_perp > 150 else QColor(0, 0, 0))
                        
                        best_perp_item.setBackground(get_color(best_perp))
                        best_perp_item.setForeground(QColor(255, 255, 255) if best_perp > 150 else QColor(0, 0, 0))
                    except (ValueError, TypeError):
                        pass
            
            summary_layout.addWidget(table)
            tab_widget.addTab(summary_tab, "Summary")
            
            # Tab 2: Training History
            history_tab = QWidget()
            history_layout = QVBoxLayout(history_tab)
            
            # Create matplotlib figure for training history
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='#2a2a3a', sharex=True)
            
            # Plot perplexity history for each model
            colors = plt.cm.tab10(np.linspace(0, 1, len(comparison)))
            has_data = False
            
            for i, (model_name, stats) in enumerate(comparison.items()):
                if "error" in stats or not stats.get("training_history"):
                    continue
                    
                history = stats["training_history"]
                has_data = True
                
                batches = [entry.get('batch_idx', idx) for idx, entry in enumerate(history)]
                perplexities = [entry.get('perplexity', float('inf')) for entry in history]
                
                ax1.plot(batches, perplexities, '-', color=colors[i], 
                        linewidth=2, label=model_name[:20] + "..." if len(model_name) > 20 else model_name)
                ax1.set_ylabel('Perplexity', color='white')
                ax1.tick_params(axis='y', colors='white')
                ax1.grid(True, alpha=0.3, color='#555555')
                
                # Plot coherence scores if available
                coherences = [entry.get('coherence_score', 0.0) for entry in history]
                ax2.plot(batches, coherences, '-', color=colors[i], linewidth=2)
                ax2.set_ylabel('Coherence Score', color='white')
                ax2.set_xlabel('Training Batch', color='white')
                ax2.tick_params(axis='both', colors='white')
                ax2.grid(True, alpha=0.3, color='#555555')
                
            if has_data:
                ax1.legend(loc='upper right', fontsize=8, facecolor='#3a3a4a', edgecolor='white')
                ax1.set_title('Model Training Comparison', color='white', fontsize=12)
                
            # Style the plots
            for ax in [ax1, ax2]:
                ax.set_facecolor('#2a2a3a')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_color('#555555')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                
            fig.tight_layout()
            
            # Embed in Qt widget
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            canvas = FigureCanvas(fig)
            history_layout.addWidget(canvas)
            
            if not has_data:
                label = QLabel("No training history available for selected models")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet("color: #aaaabb; font-style: italic;")
                history_layout.addWidget(label)
                
            tab_widget.addTab(history_tab, "Training History")
            
            layout.addWidget(tab_widget)
            
            # Add export button
            btn_layout = QHBoxLayout()
            btn_export_csv = QPushButton("Export as CSV")
            btn_export_csv.clicked.connect(lambda: self.export_comparison_csv(comparison))
            btn_export_pdf = QPushButton("Export as PDF")
            btn_export_pdf.clicked.connect(lambda: self.export_comparison_pdf(comparison, fig))
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(dialog.accept)
            
            btn_layout.addWidget(btn_export_csv)
            btn_layout.addWidget(btn_export_pdf)
            btn_layout.addStretch()
            btn_layout.addWidget(btn_close)
            
            layout.addLayout(btn_layout)
            
            dialog.setLayout(layout)
            dialog.exec()
            
        except Exception as e:
            error_msg = f"Comparison error: {str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Comparison Error", error_msg)
            self._system_log(f"Comparison error: {str(e)}", "ERROR")
            
    def export_comparison_csv(self, comparison_data):
        """Export model comparison to CSV"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Comparison Report",
            f"model_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Model Name", "Tokens Processed", "2-grams", "3-grams", "4-grams", "5-grams",
                        "Memory (MB)", "Training Time (s)", "Final Perplexity", "Best Perplexity"
                    ])
                    
                    for model_name, stats in comparison_data.items():
                        if "error" in stats:
                            writer.writerow([model_name, f"Error: {stats['error']}", "", "", "", "", "", "", "", ""])
                            continue
                            
                        ngrams = stats.get("ngrams_learned", {})
                        writer.writerow([
                            model_name,
                            stats.get('tokens_processed', 0),
                            ngrams.get('2', 0),
                            ngrams.get('3', 0),
                            ngrams.get('4', 0),
                            ngrams.get('5', 0),
                            stats.get('memory_usage', 0),
                            stats.get('training_time', 0),
                            stats.get('final_perplexity', float('inf')),
                            stats.get('best_perplexity', float('inf'))
                        ])
                        
                self._system_log(f"Comparison exported to: {file_path}", "SUCCESS")
                QMessageBox.information(self, "Export Complete", 
                                      f"Comparison report saved to:\n{file_path}")
            except Exception as e:
                self._system_log(f"Export error: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Export Error", f"Failed to export comparison:\n{str(e)}")
                
    def export_comparison_pdf(self, comparison_data, fig):
        """Export comparison figures to PDF"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Comparison Figures",
            f"model_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "PDF Files (*.pdf);;All Files (*.*)"
        )
        
        if file_path:
            try:
                fig.savefig(file_path, facecolor=fig.get_facecolor(), edgecolor='none')
                self._system_log(f"Figures exported to: {file_path}", "SUCCESS")
                QMessageBox.information(self, "Export Complete", 
                                      f"Comparison figures saved to:\n{file_path}")
            except Exception as e:
                self._system_log(f"Figure export error: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Export Error", f"Failed to export figures:\n{str(e)}")
                
    def optimize_memory(self):
        """Run comprehensive memory optimization"""
        reply = QMessageBox.question(
            self,
            "Memory Optimization",
            "This process will prune low-frequency N-Grams and optimize the spatial graph.\n"
            "It may take several minutes for large models. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                start_time = time.time()
                self._system_log("Starting comprehensive memory optimization...", "INFO")
                
                # Show progress dialog
                progress_dialog = QDialog(self)
                progress_dialog.setWindowTitle("Optimizing Memory")
                progress_dialog.setFixedSize(400, 150)
                layout = QVBoxLayout(progress_dialog)
                
                status_label = QLabel("Analyzing memory usage...")
                progress_bar = QProgressBar()
                progress_bar.setRange(0, 100)
                time_label = QLabel("Estimated time: Calculating...")
                
                layout.addWidget(status_label)
                layout.addWidget(progress_bar)
                layout.addWidget(time_label)
                
                progress_dialog.show()
                QApplication.processEvents()
                
                # Initial memory usage
                initial_memory = self.engine.memory_usage_mb
                
                # Optimization steps
                steps = [
                    ("Pruning infrequent N-Grams", 40),
                    ("Rebuilding spatial graph", 30),
                    ("Cleaning trajectory tunnels", 20),
                    ("Updating memory statistics", 10)
                ]
                
                total_progress = 0
                for step_name, step_weight in steps:
                    status_label.setText(step_name)
                    QApplication.processEvents()
                    
                    # Simulate progress within step
                    for i in range(1, 11):
                        progress_bar.setValue(total_progress + int(step_weight * i / 10))
                        elapsed = time.time() - start_time
                        estimated_total = elapsed / ((total_progress + step_weight * i / 10) / 100)
                        remaining = estimated_total - elapsed
                        time_label.setText(f"Elapsed: {int(elapsed)}s | Est. remaining: {int(remaining)}s")
                        QApplication.processEvents()
                        time.sleep(0.1)
                        
                    total_progress += step_weight
                    
                # Actually perform optimization
                self.engine._optimize_memory()
                
                # Final updates
                final_memory = self.engine.memory_usage_mb
                memory_saved = initial_memory - final_memory
                progress_bar.setValue(100)
                status_label.setText(f"Optimization complete! Saved {memory_saved:.1f} MB")
                time_label.setText(f"Total time: {int(time.time() - start_time)} seconds")
                QApplication.processEvents()
                
                # Keep dialog visible briefly
                QTimer.singleShot(1500, progress_dialog.accept)
                
                # Update UI
                self.refresh_visualization()
                self._update_status_bar()
                
                self._system_log(
                    f"Memory optimization complete. Saved {memory_saved:.1f} MB. "
                    f"New usage: {final_memory:.1f} MB",
                    "SUCCESS"
                )
                
            except Exception as e:
                error_msg = f"Optimization error: {str(e)}"
                QMessageBox.critical(self, "Optimization Error", error_msg)
                self._system_log(error_msg, "ERROR")
                
    def save_current_model(self):
        """Save the current model with validation"""
        if not self.engine.tokens_processed:
            QMessageBox.warning(self, "No Training Data", 
                              "No tokens have been processed. Please train the model first.")
            return
            
        # Suggest model name based on training data
        default_name = f"voxel_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if hasattr(self, 'current_corpus'):
            corpus_name = os.path.splitext(os.path.basename(self.current_corpus))[0]
            default_name = f"{corpus_name[:20]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        model_name, ok = QInputDialog.getText(
            self,
            "Save Model",
            "Enter model name:",
            text=default_name
        )
        
        if ok and model_name.strip():
            try:
                save_path = self.engine.save_model(model_name.strip())
                self.refresh_model_list()
                
                # Get model size
                model_size = sum(os.path.getsize(os.path.join(dirpath,filename)) 
                               for dirpath, dirnames, filenames in os.walk(save_path) 
                               for filename in filenames) / (1024 * 1024)  # MB
                
                success_msg = (
                    f"Model successfully saved to:\n{save_path}\n\n"
                    f"Training Summary:\n"
                    f"- Tokens Processed: {self.engine.tokens_processed:,}\n"
                    f"- Total N-Grams: {sum(self.engine.ngrams_learned.values()):,}\n"
                    f"- Model Size: {model_size:.1f} MB\n"
                    f"- Training Time: {self.engine.training_time:.1f} seconds"
                )
                
                self._system_log(f"Model saved: {model_name} ({model_size:.1f} MB)", "SUCCESS")
                QMessageBox.information(self, "Save Complete", success_msg)
                
            except Exception as e:
                error_msg = f"Failed to save model:\n{str(e)}"
                QMessageBox.critical(self, "Save Error", error_msg)
                self._system_log(f"Save error: {str(e)}", "ERROR")
                
    def load_model_dialog(self):
        """Open dialog to load a model from custom location"""
        model_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Model Directory",
            self.engine.model_dir
        )
        
        if model_dir and os.path.exists(model_dir):
            # Check if it's a valid model
            metadata_path = os.path.join(model_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                reply = QMessageBox.question(
                    self,
                    "Invalid Model Directory",
                    "Selected directory doesn't contain a valid VoxelPath model.\n"
                    "Would you like to scan subdirectories for models?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Scan subdirectories
                    valid_models = []
                    for root, dirs, files in os.walk(model_dir):
                        if "metadata.json" in files:
                            valid_models.append(root)
                            
                    if valid_models:
                        model_dir = valid_models[0]
                        if len(valid_models) > 1:
                            # Let user choose if multiple found
                            choice, ok = QInputDialog.getItem(
                                self,
                                "Multiple Models Found",
                                "Select model to load:",
                                [os.path.basename(p) for p in valid_models],
                                0,
                                False
                            )
                            if ok:
                                model_dir = valid_models[[os.path.basename(p) for p in valid_models].index(choice)]
                    else:
                        QMessageBox.warning(self, "No Models Found", 
                                          "No valid VoxelPath models found in the directory tree.")
                        return
                        
            # Load the model
            try:
                # Show loading dialog
                loading_dialog = QDialog(self)
                loading_dialog.setWindowTitle("Loading Model")
                loading_dialog.setFixedSize(300, 100)
                loading_layout = QVBoxLayout(loading_dialog)
                
                progress_label = QLabel("Loading model components...")
                progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                loading_layout.addWidget(progress_label)
                
                progress_bar = QProgressBar()
                progress_bar.setRange(0, 100)
                loading_layout.addWidget(progress_bar)
                
                loading_dialog.show()
                QApplication.processEvents()
                
                # Load model
                success = self.engine.load_model(model_dir)
                
                if success:
                    # Update progress
                    progress_bar.setValue(80)
                    progress_label.setText("Updating interface...")
                    QApplication.processEvents()
                    
                    # Refresh UI
                    self.refresh_visualization()
                    self.update_training_statistics({
                        'tokens_processed': self.engine.tokens_processed,
                        'ngrams_learned': self.engine.ngrams_learned,
                        'memory_usage_mb': self.engine.memory_usage_mb
                    })
                    self.refresh_model_list()
                    
                    # Get metadata
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    # Final progress update
                    progress_bar.setValue(100)
                    progress_label.setText("Model loaded successfully!")
                    QApplication.processEvents()
                    
                    # Keep dialog visible for 0.5 seconds
                    QTimer.singleShot(500, loading_dialog.accept)
                    
                    model_name = os.path.basename(model_dir)
                    self._system_log(f"Model loaded: {model_name}", "SUCCESS")
                    QMessageBox.information(self, "Load Complete", 
                                          f"Model loaded successfully:\n{model_name}")
                else:
                    loading_dialog.close()
                    self._system_log("Failed to load model", "ERROR")
                    
            except Exception as e:
                loading_dialog.close()
                error_msg = f"Failed to load model:\n{str(e)}"
                QMessageBox.critical(self, "Load Error", error_msg)
                self._system_log(f"Load error: {str(e)}", "ERROR")
                
    def export_generated_text(self):
        """Export the currently generated text with metadata"""
        if not hasattr(self, 'last_generated_text') or not self.last_generated_text:
            QMessageBox.warning(self, "No Text", "No generated text to export.")
            return
            
        default_name = f"generated_text_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Generated Text",
            default_name,
            "Text Files (*.txt);;JSON Files (*.json);;All Files (*.*)"
        )
        
        if file_path:
            try:
                # Get file extension
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.json':
                    # Export with full metadata
                    export_data = {
                        'text': self.last_generated_text,
                        'generation_parameters': {
                            'seed_text': self.txt_seed.toPlainText().strip(),
                            'temperature': self.slider_temp.value() / 100.0,
                            'top_p': self.slider_top_p.value() / 100.0,
                            'repetition_penalty': self.slider_rep_penalty.value() / 10.0,
                            'max_length': self.spin_output_length.value(),
                            'min_length': self.spin_min_length.value(),
                            'use_astar': self.cb_use_astar.isChecked(),
                            'use_momentum': self.cb_use_momentum.isChecked()
                        },
                        'metrics': self.last_generation_metrics,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'model_info': {
                            'tokens_processed': self.engine.tokens_processed,
                            'ngrams_learned': self.engine.ngrams_learned
                        }
                    }
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                        
                else:
                    # Export as text with header
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"VoxelPath Generated Text\n")
                        f.write(f"{'=' * 50}\n")
                        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Seed Text: {self.txt_seed.toPlainText().strip()}\n")
                        f.write(f"Temperature: {self.slider_temp.value() / 100.0:.2f}\n")
                        f.write(f"Top-P: {self.slider_top_p.value() / 100.0:.2f}\n")
                        f.write(f"Repetition Penalty: {self.slider_rep_penalty.value() / 10.0:.1f}\n")
                        f.write(f"Length: {len(self.last_generated_text)} characters\n")
                        
                        if hasattr(self, 'last_generation_metrics'):
                            metrics = self.last_generation_metrics
                            f.write(f"\nQuality Metrics:\n")
                            f.write(f"- Perplexity: {metrics.get('perplexity', 'N/A')}\n")
                            f.write(f"- Coherence Score: {metrics.get('coherence_score', 0.0):.2f}\n")
                            f.write(f"- Diversity Score: {metrics.get('diversity_score', 0.0):.2f}\n")
                            f.write(f"- Repetition Rate: {metrics.get('repetition_rate', 0.0) * 100:.1f}%\n")
                            
                        f.write(f"\n{'=' * 50}\n\n")
                        f.write(self.last_generated_text)
                        
                self._system_log(f"Text exported to: {file_path}", "SUCCESS")
                reply = QMessageBox.question(
                    self,
                    "Export Complete",
                    f"Generated text successfully saved to:\n{file_path}\n\n"
                    "Would you like to open the file?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
                    
            except Exception as e:
                self._system_log(f"Export error: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Export Error", f"Failed to export text:\n{str(e)}")
                
    def show_trajectory_visualization(self):
        """Show dedicated trajectory visualization window"""
        if not self.current_trajectory or len(self.current_trajectory) < 2:
            QMessageBox.warning(self, "No Trajectory", "No trajectory data available. Please generate text first.")
            return
            
        # Create new window
        traj_window = QMainWindow(self)
        traj_window.setWindowTitle("Trajectory Analysis")
        traj_window.resize(1000, 800)
        
        # Create central widget
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        # Create matplotlib figure
        fig = Figure(figsize=(10, 8), facecolor='#1a1a2a')
        canvas = FigureCanvas(fig)
        
        # Create 3D axis
        ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2a')
        
        # Plot trajectory
        trajectory = np.array(self.current_trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                color='#00ffff', linewidth=2.5, alpha=0.9)
        
        # Plot points with size based on velocity
        if hasattr(self.engine, 'geometric_momentum'):
            velocities = []
            for i in range(1, len(trajectory)):
                dx = trajectory[i, 0] - trajectory[i-1, 0]
                dy = trajectory[i, 1] - trajectory[i-1, 1]
                dz = trajectory[i, 2] - trajectory[i-1, 2]
                velocity = math.sqrt(dx*dx + dy*dy + dz*dz)
                velocities.append(velocity)
                
            # Normalize sizes
            if velocities:
                max_vel = max(velocities)
                sizes = [50 * (v / max_vel) if max_vel > 0 else 10 for v in velocities]
                scatter = ax.scatter(trajectory[1:, 0], trajectory[1:, 1], trajectory[1:, 2], 
                                   s=sizes, c=velocities, cmap='plasma', alpha=0.7)
                fig.colorbar(scatter, ax=ax, label='Velocity')
                
        # Plot start and end points
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  color='#00ff00', s=150, label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  color='#ff0000', s=150, label='End')
        
        # Set labels and title
        ax.set_xlabel('X Coordinate', color='white')
        ax.set_ylabel('Y Coordinate', color='white')
        ax.set_zlabel('Z Coordinate', color='white')
        ax.set_title('Inference Trajectory Analysis', color='white', fontsize=14)
        
        # Style the plot
        ax.grid(True, alpha=0.3, color='#555555')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#555555')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.title.set_color('white')
        
        # Add legend
        ax.legend(loc='upper right', facecolor='#2a2a3a', edgecolor='#555555')
        for text in ax.get_legend().get_texts():
            text.set_color('white')
            
        # Add statistics text box
        stats_text = (
            f"Trajectory Statistics:\n"
            f"Points: {len(trajectory)}\n"
            f"Total Distance: {self._calculate_trajectory_distance(trajectory):.1f}\n"
            f"Avg Step Size: {self._calculate_avg_step_size(trajectory):.2f}\n"
            f"Coherence Score: {self.last_generation_metrics.get('coherence_score', 0.0):.2f}"
        )
        
        props = dict(boxstyle='round', facecolor='#2a2a3a', alpha=0.8, edgecolor='#555555')
        ax.text2D(0.05, 0.95, stats_text, transform=ax.transAxes, 
                 fontsize=10, color='white', bbox=props)
        
        # Adjust layout
        fig.tight_layout()
        
        # Add to layout
        layout.addWidget(canvas)
        
        # Add control buttons
        btn_layout = QHBoxLayout()
        btn_export = QPushButton("Export Trajectory Data")
        btn_export.clicked.connect(lambda: self.export_trajectory_data(trajectory))
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(traj_window.close)
        
        btn_layout.addWidget(btn_export)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
        
        traj_window.setCentralWidget(central_widget)
        traj_window.show()
        
    def _calculate_trajectory_distance(self, trajectory: np.ndarray) -> float:
        """Calculate total Euclidean distance of trajectory"""
        total_dist = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i, 0] - trajectory[i-1, 0]
            dy = trajectory[i, 1] - trajectory[i-1, 1]
            dz = trajectory[i, 2] - trajectory[i-1, 2]
            total_dist += math.sqrt(dx*dx + dy*dy + dz*dz)
        return total_dist
        
    def _calculate_avg_step_size(self, trajectory: np.ndarray) -> float:
        """Calculate average step size in trajectory"""
        if len(trajectory) < 2:
            return 0.0
            
        total_dist = self._calculate_trajectory_distance(trajectory)
        return total_dist / (len(trajectory) - 1)
        
    def export_trajectory_data(self, trajectory: np.ndarray):
        """Export trajectory data to CSV"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Trajectory Data",
            f"trajectory_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', 'x', 'y', 'z', 'velocity'])
                    
                    # Calculate velocities
                    velocities = [0.0]  # First point has no velocity
                    for i in range(1, len(trajectory)):
                        dx = trajectory[i, 0] - trajectory[i-1, 0]
                        dy = trajectory[i, 1] - trajectory[i-1, 1]
                        dz = trajectory[i, 2] - trajectory[i-1, 2]
                        velocity = math.sqrt(dx*dx + dy*dy + dz*dz)
                        velocities.append(velocity)
                        
                    # Write data
                    for i, (point, vel) in enumerate(zip(trajectory, velocities)):
                        writer.writerow([i, point[0], point[1], point[2], vel])
                        
                self._system_log(f"Trajectory data exported to: {file_path}", "SUCCESS")
                QMessageBox.information(self, "Export Complete", 
                                      f"Trajectory data saved to:\n{file_path}")
            except Exception as e:
                self._system_log(f"Trajectory export error: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Export Error", f"Failed to export trajectory:\n{str(e)}")
                
    def open_batch_generation(self):
        """Open advanced batch generation dialog"""
        if not self.engine.tokens_processed:
            QMessageBox.warning(self, "No Model", "Please train a model before generating text.")
            return
            
        dialog = BatchGenerationDialog(self.engine, self)
        if dialog.exec():
            if dialog.generated_files:
                avg_coherence = sum(m.get('coherence_score', 0.0) for _, m in dialog.generated_files) / len(dialog.generated_files)
                avg_diversity = sum(m.get('diversity_score', 0.0) for _, m in dialog.generated_files) / len(dialog.generated_files)
                
                summary = (
                    f"Batch generation completed successfully!\n\n"
                    f"Files generated: {len(dialog.generated_files)}\n"
                    f"Average Coherence: {avg_coherence:.2f}\n"
                    f"Average Diversity: {avg_diversity:.2f}\n\n"
                    f"Output directory:\n{os.path.dirname(dialog.generated_files[0][0])}"
                )
                
                reply = QMessageBox.information(
                    self,
                    "Batch Generation Complete",
                    summary,
                    QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Ok
                )
                
                if reply == QMessageBox.StandardButton.Open:
                    output_dir = os.path.dirname(dialog.generated_files[0][0])
                    QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))
                    
    def show_about_dialog(self):
        """Show comprehensive about dialog"""
        about_html = """
        <h2>VoxelPath Production System v3.1</h2>
        <p><b>A Geometric 3D Coordinate-Mapped Trajectory Model with Hierarchical N-Gram Memory</b></p>
        
        <h3>Core Architecture</h3>
        <ul>
            <li><b>Stochastic Graph-Based Architecture</b>: Combines N-Gram statistics with geometric spatial relationships</li>
            <li><b>3D Voxel Representation</b>: Maps tokens to 3D space for spatial reasoning</li>
            <li><b>Hierarchical Memory</b>: Simultaneous 2-gram to 5-gram learning with recursive backoff</li>
            <li><b>Geometric Momentum</b>: Tracks velocity and acceleration through token space</li>
        </ul>
        
        <h3>Training & Evaluation</h3>
        <ul>
            <li>Comprehensive metrics including perplexity, coherence, and diversity scores</li>
            <li>Validation support with early stopping</li>
            <li>Memory optimization and pruning</li>
            <li>Interactive training visualization</li>
        </ul>
        
        <h3>Production Features</h3>
        <ul>
            <li>Advanced text generation with multiple sampling strategies</li>
            <li>Batch generation with parameter variation</li>
            <li>3D trajectory visualization and analysis</li>
            <li>Model comparison and benchmarking</li>
            <li>Quality analysis and optimization recommendations</li>
        </ul>
        
        <h3>System Requirements</h3>
        <ul>
            <li>Python 3.9+</li>
            <li>NumPy, Matplotlib</li>
            <li>PyQt6</li>
            <li>qdarktheme</li>
        </ul>
        
        <p><i>Copyright &copy; 2026 VoxelPath Systems. All rights reserved.</i></p>
        <p><small>For documentation and support, visit our online resources or contact the development team.</small></p>
        """
        
        dialog = QDialog(self)
        dialog.setWindowTitle("About VoxelPath")
        dialog.setMinimumSize(700, 600)
        
        layout = QVBoxLayout()
        
        # Create scrollable text area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        about_label = QLabel(about_html)
        about_label.setFont(QFont("Arial", 10))
        about_label.setWordWrap(True)
        about_label.setOpenExternalLinks(True)
        about_label.setTextFormat(Qt.TextFormat.RichText)
        about_label.setStyleSheet("color: #e0e0e0; padding: 10px;")
        
        content_layout.addWidget(about_label)
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        # Version and build info
        version_info = QLabel(
            f"Version: 3.1.0 | Build: {datetime.datetime.now().strftime('%Y%m%d')} | "
            f"Python: {sys.version.split()[0]}"
        )
        version_info.setStyleSheet("color: #888888; font-size: 9pt; padding: 5px;")
        version_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_info)
        
        # Close button
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(dialog.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        dialog.setLayout(layout)
        dialog.exec()
        
    def open_documentation(self):
        """Open comprehensive documentation"""
        doc_html = """
        <h1>VoxelPath Production System Documentation</h1>
        
        <h2>1. Getting Started</h2>
        <p><b>Initial Setup:</b></p>
        <ol>
            <li>Load a training corpus using <i>File > Load Training Corpus</i></li>
            <li>Configure training parameters (batch size, memory limits)</li>
            <li>Start training with the <i>Start Training</i> button</li>
            <li>Save your trained model using <i>File > Save Model</i></li>
        </ol>
        
        <p><b>Basic Generation:</b></p>
        <ol>
            <li>Enter seed text in the Generation tab</li>
            <li>Adjust generation parameters (temperature, top-p, repetition penalty)</li>
            <li>Click <i>Generate Text</i> to produce output</li>
            <li>Analyze quality using the <i>Analyze Generated Text</i> button</li>
        </ol>
        
        <h2>2. Advanced Training</h2>
        <p><b>Training Best Practices:</b></p>
        <ul>
            <li><b>Corpus Size:</b> Start with 1-10MB of clean text for initial training</li>
            <li><b>Batch Size:</b> Larger batches (50K-200K characters) improve efficiency</li>
            <li><b>Validation:</b> Always use a validation corpus (10% of training data)</li>
            <li><b>Early Stopping:</b> Set patience to 3-5 batches to prevent overfitting</li>
        </ul>
        
        <p><b>Key Metrics:</b></p>
        <ul>
            <li><b>Perplexity:</b> Lower values indicate better model fit (target < 150)</li>
            <li><b>Coherence Score:</b> Measures sentence-to-sentence flow (target > 0.6)</li>
            <li><b>Diversity Score:</b> Measures vocabulary variety (target > 0.4)</li>
            <li><b>Memory Usage:</b> Monitor to prevent system slowdowns</li>
        </ul>
        
        <h2>3. Advanced Generation</h2>
        <p><b>Parameter Guide:</b></p>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Parameter</th>
                <th>Low Value</th>
                <th>High Value</th>
                <th>Effect</th>
            </tr>
            <tr>
                <td>Temperature</td>
                <td>0.5-0.7</td>
                <td>0.9-1.2</td>
                <td>Higher = more creative, less predictable</td>
            </tr>
            <tr>
                <td>Top-P</td>
                <td>0.7-0.8</td>
                <td>0.9-0.95</td>
                <td>Higher = more diverse sampling</td>
            </tr>
            <tr>
                <td>Repetition Penalty</td>
                <td>1.0-1.2</td>
                <td>1.3-2.0</td>
                <td>Higher = less repetition</td>
            </tr>
        </table>
        
        <p><b>Advanced Features:</b></p>
        <ul>
            <li><b>A* Pathfinding:</b> Enables geometric trajectory search for coherent output</li>
            <li><b>Geometric Momentum:</b> Maintains consistent writing style and pace</li>
            <li><b>Trajectory Tunnels:</b> Creates shortcuts for common patterns</li>
            <li><b>Adaptive Temperature:</b> Automatically adjusts based on context complexity</li>
        </ul>
        
        <h2>4. Model Management</h2>
        <p><b>Best Practices:</b></p>
        <ul>
            <li>Save models after significant training milestones</li>
            <li>Use descriptive names including corpus source and date</li>
            <li>Compare models using the comparison tool to select best performers</li>
            <li>Regularly optimize memory to maintain performance</li>
        </ul>
        
        <h2>5. Troubleshooting</h2>
        <p><b>Common Issues:</b></p>
        <ul>
            <li><b>Poor Coherence:</b> Lower temperature, increase repetition penalty, add more training data</li>
            <li><b>High Repetition:</b> Increase repetition penalty, raise temperature slightly</li>
            <li><b>Memory Issues:</b> Reduce batch size, enable aggressive memory optimization</li>
            <li><b>Slow Training:</b> Increase batch size, reduce corpus size, close other applications</li>
        </ul>
        
        <h2>6. API Reference</h2>
        <p>For programmatic use, see the source code documentation for:</p>
        <ul>
            <li><code>VoxelPathProductionEngine</code> class methods</li>
            <li><code>IntelligentTokenizer</code> capabilities</li>
            <li><code>TrainingWorker</code> and <code>GenerationWorker</code> threading</li>
        </ul>
        
        <p><i>For more detailed examples and use cases, visit our online documentation portal.</i></p>
        """
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Comprehensive Documentation")
        dialog.setMinimumSize(900, 700)
        
        layout = QVBoxLayout()
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Documentation tab
        doc_tab = QWidget()
        doc_layout = QVBoxLayout(doc_tab)
        
        # Create scrollable text area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        doc_label = QLabel(doc_html)
        doc_label.setFont(QFont("Arial", 10))
        doc_label.setWordWrap(True)
        doc_label.setOpenExternalLinks(True)
        doc_label.setTextFormat(Qt.TextFormat.RichText)
        doc_label.setStyleSheet("color: #e0e0e0; padding: 15px; line-height: 1.5;")
        
        content_layout.addWidget(doc_label)
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        doc_layout.addWidget(scroll_area)
        
        tab_widget.addTab(doc_tab, "Documentation")
        
        # Quick Reference tab
        ref_tab = QWidget()
        ref_layout = QVBoxLayout(ref_tab)
        
        quick_ref = """
        <h2>Quick Reference Guide</h2>
        
        <h3>Training Shortcuts</h3>
        <ul>
            <li><b>Ctrl+T</b>: Start training</li>
            <li><b>Ctrl+S</b>: Save model</li>
            <li><b>Ctrl+O</b>: Load model</li>
            <li><b>F5</b>: Refresh model list</li>
        </ul>
        
        <h3>Generation Shortcuts</h3>
        <ul>
            <li><b>Ctrl+G</b>: Generate text</li>
            <li><b>Ctrl+A</b>: Analyze text</li>
            <li><b>F9</b>: Clear output</li>
        </ul>
        
        <h3>Recommended Parameter Sets</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Use Case</th>
                <th>Temperature</th>
                <th>Top-P</th>
                <th>Rep Penalty</th>
            </tr>
            <tr>
                <td>Factual/Technical</td>
                <td>0.6</td>
                <td>0.8</td>
                <td>1.3</td>
            </tr>
            <tr>
                <td>Creative Writing</td>
                <td>0.9</td>
                <td>0.92</td>
                <td>1.1</td>
            </tr>
            <tr>
                <td>Poetry</td>
                <td>1.1</td>
                <td>0.95</td>
                <td>1.0</td>
            </tr>
            <tr>
                <td>Code Generation</td>
                <td>0.4</td>
                <td>0.7</td>
                <td>1.5</td>
            </tr>
        </table>
        
        <h3>Memory Management</h3>
        <ul>
            <li>Models under 1GB: No optimization needed</li>
            <li>Models 1-3GB: Optimize weekly</li>
            <li>Models over 3GB: Optimize daily and consider pruning</li>
        </ul>
        """
        
        ref_label = QLabel(quick_ref)
        ref_label.setFont(QFont("Arial", 10))
        ref_label.setWordWrap(True)
        ref_label.setTextFormat(Qt.TextFormat.RichText)
        ref_label.setStyleSheet("color: #e0e0e0; padding: 15px;")
        
        ref_layout.addWidget(ref_label)
        tab_widget.addTab(ref_tab, "Quick Reference")
        
        layout.addWidget(tab_widget)
        
        # Close button
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Close")
        btn_ok.clicked.connect(dialog.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        dialog.setLayout(layout)
        dialog.exec()
        
    def show_tutorial(self):
        """Show interactive tutorial"""
        tutorial_steps = [
            {
                'title': 'Welcome to VoxelPath',
                'content': 'This interactive tutorial will guide you through the key features of the VoxelPath Production System.'
            },
            {
                'title': 'Step 1: Loading Training Data',
                'content': 'Start by loading a training corpus. Go to File > Load Training Corpus and select a text file. For best results, use clean text files between 1MB and 100MB.'
            },
            {
                'title': 'Step 2: Training the Model',
                'content': 'After loading your corpus, click the "Start Training" button. Monitor progress in the Training Progress panel. The system will show real-time metrics like perplexity and N-gram counts.'
            },
            {
                'title': 'Step 3: Generating Text',
                'content': 'Once trained, go to the Generation tab. Enter seed text and adjust parameters like temperature and repetition penalty. Click "Generate Text" to create output based on your model.'
            },
            {
                'title': 'Step 4: Analyzing Results',
                'content': 'Use the "Analyze Generated Text" button to get detailed quality metrics. The system will provide coherence scores, diversity metrics, and optimization recommendations.'
            },
            {
                'title': 'Step 5: Advanced Features',
                'content': 'Explore advanced features like 3D trajectory visualization, batch generation, and model comparison. These tools help you understand and improve your model\'s performance.'
            },
            {
                'title': 'Step 6: Model Management',
                'content': 'Save your trained models regularly. Use the Models tab to compare different versions and select the best performer for your needs.'
            },
            {
                'title': 'Tutorial Complete',
                'content': 'You\'re now ready to use VoxelPath for advanced text generation tasks! For more detailed guidance, consult the documentation or visit our support portal.'
            }
        ]
        
        # Create tutorial dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Interactive Tutorial")
        dialog.setMinimumSize(700, 500)
        
        layout = QVBoxLayout()
        
        # Step title
        title_label = QLabel(tutorial_steps[0]['title'])
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #00ffff;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Step content
        content_label = QLabel(tutorial_steps[0]['content'])
        content_label.setFont(QFont("Arial", 11))
        content_label.setWordWrap(True)
        content_label.setStyleSheet("padding: 15px; line-height: 1.5;")
        layout.addWidget(content_label)
        
        # Navigation buttons
        btn_layout = QHBoxLayout()
        
        btn_prev = QPushButton("Previous")
        btn_prev.setEnabled(False)
        btn_next = QPushButton("Next")
        btn_finish = QPushButton("Finish")
        btn_finish.setVisible(False)
        
        btn_layout.addWidget(btn_prev)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_next)
        btn_layout.addWidget(btn_finish)
        
        layout.addLayout(btn_layout)
        
        # Current step tracker
        current_step = [0]
        
        def update_step(step_index):
            current_step[0] = step_index
            step = tutorial_steps[step_index]
            
            title_label.setText(step['title'])
            content_label.setText(step['content'])
            
            btn_prev.setEnabled(step_index > 0)
            btn_next.setVisible(step_index < len(tutorial_steps) - 1)
            btn_finish.setVisible(step_index == len(tutorial_steps) - 1)
            
        def next_step():
            if current_step[0] < len(tutorial_steps) - 1:
                update_step(current_step[0] + 1)
                
        def prev_step():
            if current_step[0] > 0:
                update_step(current_step[0] - 1)
                
        btn_next.clicked.connect(next_step)
        btn_prev.clicked.connect(prev_step)
        btn_finish.clicked.connect(dialog.accept)
        
        dialog.setLayout(layout)
        dialog.exec()
        
    def closeEvent(self, event):
        """Handle application close event with proper cleanup"""
        # Stop any running workers
        workers_to_stop = []
        
        if self.training_worker and self.training_worker.isRunning():
            workers_to_stop.append(('training', self.training_worker))
            
        if self.generation_worker and self.generation_worker.isRunning():
            workers_to_stop.append(('generation', self.generation_worker))
            
        if workers_to_stop:
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("Stopping Processes")
            progress_dialog.setFixedSize(350, 120)
            dialog_layout = QVBoxLayout(progress_dialog)
            
            status_label = QLabel("Stopping active processes...")
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            
            dialog_layout.addWidget(status_label)
            dialog_layout.addWidget(progress_bar)
            
            progress_dialog.show()
            QApplication.processEvents()
            
            total_workers = len(workers_to_stop)
            for i, (worker_type, worker) in enumerate(workers_to_stop):
                progress = int((i / total_workers) * 100)
                progress_bar.setValue(progress)
                status_label.setText(f"Stopping {worker_type} process...")
                QApplication.processEvents()
                
                worker.stop()
                # Wait with timeout
                if not worker.wait(3000):  # 3 seconds timeout
                    self._system_log(f"Could not stop {worker_type} process gracefully", "WARNING")
                    
            progress_bar.setValue(100)
            status_label.setText("All processes stopped")
            QApplication.processEvents()
            QTimer.singleShot(500, progress_dialog.accept)
            
        # Ask about saving model
        if self.engine.tokens_processed > 0:
            reply = QMessageBox.question(
                self,
                "Save Model Before Exit",
                "You have an active model with training data. Would you like to save it before exiting?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
                
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    # Suggest a name based on current time
                    default_name = f"auto_save_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    model_name, ok = QInputDialog.getText(
                        self,
                        "Save Model",
                        "Enter model name for auto-save:",
                        text=default_name
                    )
                    
                    if ok and model_name:
                        save_path = self.engine.save_model(model_name)
                        self._system_log(f"Auto-saved model to: {save_path}", "SUCCESS")
                except Exception as e:
                    self._system_log(f"Auto-save failed: {str(e)}", "ERROR")
                    
        # Clean up matplotlib figures
        plt.close('all')
        
        # Accept the close event
        event.accept()
        QApplication.quit()
        
# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point with exception handling"""
    try:
        app = QApplication(sys.argv)
        
        # Set application style
        app.setStyle('Fusion')
        app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
        
        # Set application icon and metadata
        app.setApplicationName("VoxelPath Production System")
        app.setApplicationVersion("3.1.0")
        app.setOrganizationName("VoxelPath Systems")
        app.setOrganizationDomain("voxelpath.systems")
        
        # Create and show main window
        window = VoxelPathProductionUI()
        window.show()
        
        # Start application
        sys.exit(app.exec())
        
    except Exception as e:
        import traceback
        error_msg = f"Application crashed: {str(e)}\n{traceback.format_exc()}"
        
        # Try to show error dialog
        try:
            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "Critical Error", 
                               "VoxelPath encountered a critical error and must close.\n\n"
                               f"Error details:\n{error_msg}")
        except:
            # Fallback to console
            print(error_msg, file=sys.stderr)
            
        sys.exit(1)
        
if __name__ == "__main__":
    # Set up exception handling for threads
    import threading
    def handle_exception(exc_type, exc_value, exc_traceback):
        import traceback
        error_msg = f"Uncaught exception: {exc_type.__name__}: {exc_value}\n"
        error_msg += "\n".join(traceback.format_tb(exc_traceback))
        print(error_msg, file=sys.stderr)
        
        # Try to show error dialog if application is running
        try:
            if QApplication.instance():
                QMessageBox.critical(None, "Uncaught Exception", 
                                   "An unhandled error occurred:\n\n" + error_msg)
        except:
            pass
            
    sys.excepthook = handle_exception
    
    # Start main application
    main()