/**
 * API client configuration
 */

export const API_CONFIG = {
    baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
    timeout: 30000, // 30 seconds
} as const;

/**
 * API endpoints
 */
export const API_ENDPOINTS = {
    // Generation
    GENERATE: '/generation/generate',
    CHAT: '/generation/chat',
    BATCH: '/generation/batch',
    ANALYZE: '/generation/analyze',

    // Training
    TRAINING_START: '/training/start',
    TRAINING_STATUS: '/training/status',
    TRAINING_CANCEL: '/training/cancel',

    // Models
    MODELS_LIST: '/models/list',
    MODELS_SAVE: '/models/save',
    MODELS_LOAD: '/models/load',
    MODELS_DELETE: '/models',
    MODELS_EXPORT: '/models/export',
    MODELS_IMPORT: '/models/import',
    MODELS_LOADED: '/models/loaded',

    // RAG
    RAG_INGEST: '/rag/ingest',
    RAG_QUERY: '/rag/query',
    RAG_DOCUMENTS: '/rag/documents',
    RAG_STATS: '/rag/stats',

    // System
    STATUS: '/status',
    HEALTH: '/health',
} as const;
