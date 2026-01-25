import datetime
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from enum import Enum
from collections import defaultdict
import math
import numpy as np

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
            self.velocity[0] / time_delta,
            self.velocity[1] / time_delta,
            self.velocity[2] / time_delta
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
    training_loss: float = 0.0  # Online training loss (NLL)
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
            'training_loss': self.training_loss,
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
        metrics.training_loss = data.get('training_loss', 0.0)
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
    flesch_reading_ease: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'perplexity': self.perplexity,
            'coherence_score': self.coherence_score,
            'diversity_score': self.diversity_score,
            'repetition_rate': self.repetition_rate,
            'avg_word_length': self.avg_word_length,
            'sentence_complexity': self.sentence_complexity,
            'sentiment': self.sentiment,
            'topic_consistency': self.topic_consistency,
            'grammatical_correctness': self.grammatical_correctness,
            'flesch_reading_ease': self.flesch_reading_ease
        }
