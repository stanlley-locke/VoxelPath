from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum

# ============================================================================
# Generation Models
# ============================================================================

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.85
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    stop_sequences: Optional[List[str]] = None
    use_momentum: bool = True

class GenerationResponse(BaseModel):
    text: str
    metrics: Dict
    trajectory: List[tuple]

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_length: int = 200
    temperature: float = 0.85
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    use_momentum: bool = True

class ChatResponse(BaseModel):
    message: ChatMessage
    metrics: Dict

class BatchGenerationRequest(BaseModel):
    prompts: List[str]
    max_length: int = 200
    temperature: float = 0.85
    top_p: float = 0.9
    repetition_penalty: float = 1.2

class BatchGenerationResponse(BaseModel):
    results: List[Dict[str, Any]]  # [{prompt, text, metrics}, ...]

class TextAnalysisRequest(BaseModel):
    text: str

class TextAnalysisResponse(BaseModel):
    coherence_score: float
    diversity_score: float
    perplexity: float
    unique_tokens: int
    total_tokens: int
    avg_token_length: float

# ============================================================================
# Training Models
# ============================================================================

class TrainingConfig(BaseModel):
    batch_size: int = 10000
    validation_split: float = 0.1
    max_ngram_level: int = 5
    checkpoint_interval: int = 5
    early_stopping_patience: int = 5
    detailed_logging: bool = True
    use_label_smoothing: bool = True

class TrainingRequest(BaseModel):
    corpus: str
    config: Optional[TrainingConfig] = None
    validation_corpus: Optional[str] = None

class TrainingResponse(BaseModel):
    status: str
    metrics: Dict

class TrainingStatus(BaseModel):
    is_training: bool
    progress_percent: float
    current_batch: int
    total_batches: int
    metrics: Dict

# ============================================================================
# Model Management
# ============================================================================

class ModelInfo(BaseModel):
    name: str
    size_mb: float
    created: str
    metrics: Dict

class ModelListResponse(BaseModel):
    models: List[ModelInfo]

class ModelSaveRequest(BaseModel):
    name: str
    metadata: Optional[Dict] = None

class ModelSaveResponse(BaseModel):
    status: str
    path: str

class ModelLoadedInfo(BaseModel):
    name: str
    ngrams_count: int
    memory_mb: float
    loaded_at: str

# ============================================================================
# RAG Models
# ============================================================================

class RAGIngestRequest(BaseModel):
    filename: str
    content: str

class RAGResponse(BaseModel):
    status: str
    chunks_added: int
    total_documents: int

class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 5

class RAGQueryResponse(BaseModel):
    results: List[Dict[str, Any]]  # [{content, score, source}, ...]

class RAGDocument(BaseModel):
    id: str
    source: str
    content: str
    created: str

class RAGDocumentsResponse(BaseModel):
    documents: List[RAGDocument]

class RAGStatsResponse(BaseModel):
    total_documents: int
    total_vectors: int
    storage_type: str  # "chromadb" or "simple"

# ============================================================================
# Analytics Models
# ============================================================================

class AnalyticsResponse(BaseModel):
    training_history: List[Dict]
    total_tokens: int
    uptime: float

class MetricsHistoryResponse(BaseModel):
    epochs: List[int]
    loss: List[float]
    perplexity: List[float]
    ngrams: List[Dict[str, int]]

class NGramBreakdownResponse(BaseModel):
    level_2: int
    level_3: int
    level_4: int
    level_5: int
    total: int

class VocabularyStatsResponse(BaseModel):
    unique_tokens: int
    total_tokens: int
    avg_token_length: float

# ============================================================================
# System Models
# ============================================================================

class SystemStatus(BaseModel):
    is_training: bool
    ngrams_learned: int
    memory_usage_mb: float
    uptime_seconds: float

class HealthResponse(BaseModel):
    status: str  # "healthy", "degraded", "unhealthy"
    components: Dict[str, str]  # {engine: "ok", storage: "ok", ...}

class SystemLogsRequest(BaseModel):
    lines: int = 100
    level: Optional[str] = None  # "INFO", "ERROR", etc.

class SystemLogsResponse(BaseModel):
    logs: List[str]

