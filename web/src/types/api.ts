/**
 * API Types - Generated from VoxelLM API
 */

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
}

export interface ChatRequest {
    messages: ChatMessage[];
    max_length?: number;
    temperature?: number;
    top_p?: number;
    repetition_penalty?: number;
    use_momentum?: boolean;
}

export interface ChatResponse {
    message: ChatMessage;
    metrics: Record<string, MetricValue>;
}

export interface TrainingConfig {
    batch_size?: number;
    validation_split?: number;
    max_ngram_level?: number;
    checkpoint_interval?: number;
    early_stopping_patience?: number;
    detailed_logging?: boolean;
    use_label_smoothing?: boolean;
}

export interface TrainingRequest {
    corpus: string;
    config?: TrainingConfig;
    validation_corpus?: string;
}

export interface TrainingStatus {
    is_training: boolean;
    progress_percent: number;
    current_batch: number;
    total_batches: number;
    metrics: Record<string, MetricValue>;
}

export interface ModelInfo {
    name: string;
    size_mb: number;
    created: string;
    metrics: Record<string, MetricValue>;
}

export type MetricValue = string | number | boolean | null;

export interface SystemStatus {
    is_training: boolean;
    ngrams_learned: number;
    memory_usage_mb: number;
    uptime_seconds: number;
}

export interface RAGDocument {
    id: string;
    source: string;
    content: string;
    created: string;
}

export interface RAGQueryRequest {
    query: string;
    top_k?: number;
}
