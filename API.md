# VoxelLM API Documentation

Complete REST API reference for VoxelLM server.

## Base URL

```
http://localhost:8000
```

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Authentication

Currently, no authentication is required. All endpoints are publicly accessible.

---

## Generation Endpoints

### Generate Text

Generate text continuation from a prompt.

**Endpoint**: `POST /generation/generate`

**Request Body**:
```json
{
  "prompt": "Once upon a time",
  "max_length": 100,
  "temperature": 0.85,
  "use_voxel": true,
  "use_neural": true
}
```

**Response**:
```json
{
  "generated_text": "Once upon a time in a distant land...",
  "prompt": "Once upon a time",
  "stats": {
    "length": 45,
    "time_seconds": 0.23
  }
}
```

### Chat Interface

Multi-turn conversation with context.

**Endpoint**: `POST /generation/chat`

**Request Body**:
```json
{
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
  ],
  "temperature": 0.85,
  "max_length": 150
}
```

**Response**:
```json
{
  "message": {
    "role": "assistant",
    "content": "I'm doing well, thank you for asking!"
  },
  "context_used": 3
}
```

### Batch Generation

Generate multiple outputs from a prompt.

**Endpoint**: `POST /generation/batch`

**Request Body**:
```json
{
  "prompt": "The future of AI is",
  "num_generations": 5,
  "max_length": 50,
  "temperature": 1.0
}
```

**Response**:
```json
{
  "generations": [
    "The future of AI is bright and promising...",
    "The future of AI is uncertain but exciting...",
    "The future of AI is transformative..."
  ]
}
```

---

## Training Endpoints

### Start Training

Begin training on a corpus.

**Endpoint**: `POST /training/start`

**Request Body**:
```json
{
  "corpus": "Training text goes here...",
  "validation_corpus": "Optional validation text...",
  "config": {
    "batch_size": 10000,
    "validation_split": 0.1,
    "max_ngram_level": 5,
    "detailed_logging": true
  }
}
```

**Response**:
```json
{
  "status": "started",
  "corpus_size": 5560658,
  "estimated_batches": 114
}
```

### Training Status

Get current training progress.

**Endpoint**: `GET /training/status`

**Response**:
```json
{
  "is_training": true,
  "progress_percent": 45.6,
  "current_batch": 52,
  "total_batches": 114,
  "metrics": {
    "loss": 1.8234,
    "perplexity": 1.12,
    "tokens_per_second": 3250
  }
}
```

### Cancel Training

Stop ongoing training.

**Endpoint**: `POST /training/cancel`

**Response**:
```json
{
  "status": "cancelled",
  "batches_completed": 52
}
```

---

## Model Management

### List Models

Get all saved models.

**Endpoint**: `GET /models/list`

**Response**:
```json
{
  "models": [
    {
      "name": "shakespeare_v1",
      "size_mb": 12.4,
      "created": "2026-01-25T10:30:00Z"
    },
    {
      "name": "my_corpus",
      "size_mb": 8.2,
      "created": "2026-01-24T15:20:00Z"
    }
  ]
}
```

### Save Model

Save current model state.

**Endpoint**: `POST /models/save`

**Request Body**:
```json
{
  "name": "my_trained_model"
}
```

**Response**:
```json
{
  "status": "saved",
  "name": "my_trained_model",
  "path": "models/my_trained_model.pkl",
  "size_mb": 8.4
}
```

### Load Model

Load a saved model.

**Endpoint**: `POST /models/load/{model_name}`

**Parameters**:
- `model_name` (path): Name of model to load

**Response**:
```json
{
  "status": "loaded",
  "name": "my_trained_model",
  "ngrams_count": 145230
}
```

### Export Model

Download model file.

**Endpoint**: `GET /models/export/{model_name}`

**Parameters**:
- `model_name` (path): Name of model to export

**Response**: Binary file download (application/octet-stream)

---

## RAG Endpoints

### Ingest Documents

Add documents to RAG system.

**Endpoint**: `POST /rag/ingest`

**Request Body**:
```json
{
  "documents": [
    {
      "content": "Document text content...",
      "metadata": {
        "source": "doc1.txt",
        "author": "John Doe"
      }
    }
  ],
  "chunk_size": 500
}
```

**Response**:
```json
{
  "status": "ingested",
  "documents_added": 1,
  "chunks_created": 12
}
```

### Query Documents

Retrieve relevant documents.

**Endpoint**: `POST /rag/query`

**Request Body**:
```json
{
  "query": "How does VoxelLM work?",
  "top_k": 3
}
```

**Response**:
```json
{
  "results": [
    {
      "content": "VoxelLM uses 3D coordinates...",
      "score": 0.92,
      "metadata": {"source": "intro.txt"}
    },
    {
      "content": "Training involves spatial density...",
      "score": 0.88,
      "metadata": {"source": "training.txt"}
    }
  ]
}
```

### RAG-Enhanced Generation

Generate with document context.

**Endpoint**: `POST /rag/generate`

**Request Body**:
```json
{
  "prompt": "Explain VoxelLM training",
  "top_k_docs": 3,
  "max_length": 200
}
```

**Response**:
```json
{
  "generated_text": "VoxelLM training uses a sliding window...",
  "context_docs": [
    {"content": "...", "score": 0.92}
  ]
}
```

### Clear RAG Storage

Remove all documents.

**Endpoint**: `POST /rag/clear`

**Response**:
```json
{
  "status": "cleared",
  "documents_removed": 45,
  "chunks_removed": 234
}
```

---

## System Endpoints

### System Status

Get current system state.

**Endpoint**: `GET /status`

**Response**:
```json
{
  "ngrams_learned": 145230,
  "memory_usage_mb": 8.4,
  "uptime_seconds": 3600,
  "is_training": false,
  "version": "2.0.0"
}
```

### Health Check

Simple health check endpoint.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-25T12:00:00Z"
}
```

---

## Error Responses

All endpoints may return error responses:

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 404 Not Found
```json
{
  "detail": "Model 'nonexistent' not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error occurred"
}
```

---

## Rate Limiting

Currently no rate limiting is implemented. For production deployment, consider:

- Request rate limits per IP
- Concurrent training limits
- Maximum corpus size limits

## CORS

CORS is enabled for all origins in development. Configure appropriately for production.

## WebSocket Support

Future versions may include WebSocket endpoints for:
- Real-time training progress
- Streaming generation
- Live chat updates

---

## Code Examples

### Python

```python
import requests

# Generate text
response = requests.post('http://localhost:8000/generation/generate', json={
    'prompt': 'Hello world',
    'max_length': 50
})
print(response.json()['generated_text'])

# Start training
requests.post('http://localhost:8000/training/start', json={
    'corpus': 'Training data...',
    'config': {'batch_size': 10000}
})

# Check status
status = requests.get('http://localhost:8000/training/status').json()
print(f"Progress: {status['progress_percent']}%")
```

### JavaScript/TypeScript

```typescript
// Generate text
const response = await fetch('http://localhost:8000/generation/generate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    prompt: 'Hello world',
    max_length: 50
  })
});
const data = await response.json();
console.log(data.generated_text);

// Chat
const chatResponse = await fetch('http://localhost:8000/generation/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    messages: [{role: 'user', content: 'Hello!'}]
  })
});
const chat = await chatResponse.json();
console.log(chat.message.content);
```

### cURL

```bash
# Generate
curl -X POST http://localhost:8000/generation/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_length": 50}'

# Train
curl -X POST http://localhost:8000/training/start \
  -H "Content-Type: application/json" \
  -d '{"corpus": "Training text..."}'

# Status
curl http://localhost:8000/training/status
```

---

**For more details, visit the interactive documentation at http://localhost:8000/docs**
