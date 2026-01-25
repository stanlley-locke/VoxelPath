# VoxelLM Core Engine & GUI

The core VoxelLM implementation with geometric 3D coordinate-based language modeling and PyQt6 GUI application.

## Overview

This package contains the complete VoxelLM engine implementation:
- **Voxel Engine**: 3D coordinate-based pattern recognition
- **Neural Scorer**: Hybrid n-gram probability estimation
- **RAG System**: Document retrieval and context augmentation
- **Storage Engine**: SQLite-based voxel persistence
- **GUI Application**: PyQt6 desktop interface
- **FastAPI Server**: RESTful API for web integration

## Quick Start

### Command Line Interface

```bash
# Run interactive CLI
python -m voxellm.main

# Start API server
python -m voxellm.server_main
```

### GUI Application

```bash
# Linux/Mac
python -m voxellm.main

# Windows
pythonw -m voxellm.main
```

The GUI provides:
- Training interface with configuration
- Real-time generation
- Model saving/loading
- RAG document management
- Chat interface

## Architecture

### Core Components

```
voxellm/
├── core/                    # Core engine modules
│   ├── engine.py           # VoxelEngine - 3D coordinate system
│   ├── ensemble.py         # Hybrid voxel + neural scoring
│   ├── rag.py              # RAG document retrieval
│   ├── storage.py          # SQLite voxel persistence
│   ├── tokenizer.py        # Character/word tokenization
│   └── structures.py       # Data structures
├── ui/                      # PyQt6 GUI
│   ├── main_window.py      # Main application window
│   ├── widgets.py          # Custom widgets
│   ├── dialogs/            # Dialog windows
│   └── styles.py           # UI styling
├── server/                  # FastAPI server
│   ├── app.py              # FastAPI application
│   ├── routers/            # API endpoints
│   └── models.py           # Pydantic models
├── workers/                 # Background workers
│   ├── training_worker.py  # Training thread
│   └── generation_worker.py # Generation thread
└── main.py                  # CLI/GUI entry point
```

## VoxelEngine API

### Basic Usage

```python
from voxellm.core.engine import VoxelEngine

# Initialize engine
engine = VoxelEngine()

# Train on text
corpus = "The quick brown fox jumps over the lazy dog."
engine.train(corpus)

# Generate text
result = engine.generate("The quick", max_length=20)
print(result)  # "The quick brown fox..."

# Save model
engine.save_model("my_model")

# Load model
engine.load_model("my_model")
```

### Advanced Features

```python
# Hybrid scoring (voxel + neural)
from voxellm.core.ensemble import HybridEnsemble

ensemble = HybridEnsemble(
    voxel_weight=0.7,
    neural_weight=0.3
)
ensemble.train(corpus)
result = ensemble.generate("Hello", max_length=50)

# RAG integration
from voxellm.core.rag import SimpleVectorStore

rag = SimpleVectorStore()
documents = [
    {"content": "VoxelLM uses 3D coordinates...", "source": "doc1.txt"}
]
rag.ingest_documents(documents)

# Query with context
context = rag.query("How does VoxelLM work?", top_k=3)
```

## Training

### Configuration Options

```python
config = {
    "batch_size": 10000,           # Tokens per batch
    "validation_split": 0.1,       # Validation %
    "max_ngram_level": 5,          # N-gram depth (2-10)
    "detailed_logging": True,      # Verbose output
    "momentum_weight": 0.5,        # Trajectory bias
    "temperature": 0.85            # Generation randomness
}

engine.train(corpus, config=config)
```

### Training Callbacks

```python
def on_batch_complete(batch_num, total_batches, metrics):
    print(f"Batch {batch_num}/{total_batches}: Loss={metrics['loss']}")

engine.train(
    corpus, 
    on_batch=on_batch_complete
)
```

## GUI Features

### Main Window

The GUI provides an intuitive interface for:

**Training Tab:**
- Load corpus from file or paste text
- Configure training parameters with sliders
- Real-time progress bar and metrics
- Auto-save checkpoints

**Generation Tab:**
- Simple prompt input
- Adjustable temperature and length
- Real-time streaming output
- Copy/export results

**Chat Tab:**
- Multi-turn conversations
- Message history
- Context-aware responses
- Export chat logs

**RAG Tab:**
- Upload documents
- Query interface
- Context display
- Document management

**Settings Tab:**
- Model save/load
- Export model files
- Database optimization
- Logging configuration

### Keyboard Shortcuts

- `Ctrl+T` - New training session
- `Ctrl+G` - Generate text
- `Ctrl+S` - Save model
- `Ctrl+L` - Load model
- `Ctrl+Q` - Quit application

## Storage System

### Database Schema

VoxelLM uses SQLite for persistent storage:

```sql
-- Voxel density table
CREATE TABLE voxels (
    x INTEGER,
    y INTEGER,
    z INTEGER,
    density INTEGER,
    PRIMARY KEY (x, y, z)
);

-- N-gram transitions
CREATE TABLE ngrams (
    prefix TEXT,
    next_char TEXT,
    count INTEGER,
    level INTEGER
);

-- Training metadata
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

### Model Files

Saved models contain:
- `voxel_tensor.npy` - NumPy voxel grid
- `ngram_memory.pkl.gz` - N-gram statistics
- `spatial_graph.pkl.gz` - Coordinate adjacency
- `trajectory_tunnels.pkl.gz` - Macro-jump shortcuts
- `training_history.json` - Training metrics
- `metadata.json` - Model configuration

## API Server

### Starting the Server

```bash
python -m voxellm.server_main
```

Server runs at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### Example Endpoints

```python
import requests

# Generate text
response = requests.post('http://localhost:8000/generation/generate', json={
    'prompt': 'Once upon a time',
    'max_length': 100
})

# Train model
requests.post('http://localhost:8000/training/start', json={
    'corpus': 'Training text...',
    'config': {'batch_size': 10000}
})

# Get status
status = requests.get('http://localhost:8000/training/status').json()
```

See [API.md](../API.md) for complete endpoint documentation.

## Performance

### Optimization Tips

1. **Batch Size**: Larger batches (20K-50K) reduce overhead
2. **N-gram Level**: Lower levels (3-5) are faster
3. **Database**: Enable WAL mode for concurrency
4. **Memory**: Entire model fits in ~8-50 MB depending on corpus

### Benchmarks

| Corpus Size | Training Time | Memory Usage | Generation Speed |
|-------------|---------------|--------------|------------------|
| 100KB       | ~2s           | 8 MB         | 500 tok/s        |
| 1MB         | ~15s          | 15 MB        | 450 tok/s        |
| 10MB        | ~2min         | 45 MB        | 400 tok/s        |

*Tested on: Intel i7, 16GB RAM, SSD*

## Testing

```bash
# Run all tests
pytest voxellm/tests/

# Run specific test
python voxellm/tests/verify_db_engine.py

# Test GUI components
python voxellm/tests/verify_help_ui.py
```

## Development

### Adding Custom Scorers

```python
from voxellm.core.ensemble import BaseScorer

class MyCustomScorer(BaseScorer):
    def score(self, context: str, candidates: list) -> dict:
        # Return {char: probability}
        scores = {}
        for char in candidates:
            scores[char] = self.calculate_prob(context, char)
        return scores

# Use in ensemble
ensemble = HybridEnsemble(custom_scorer=MyCustomScorer())
```

### Extending the GUI

```python
from voxellm.ui.main_window import VoxelLMGUI
from PyQt6.QtWidgets import QWidget

class MyCustomWidget(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.setup_ui()

# Add to main window
gui = VoxelLMGUI()
gui.add_custom_tab("My Feature", MyCustomWidget(gui.engine))
```

## Troubleshooting

### Common Issues

**1. Import errors**
```bash
pip install -r requirements.txt
```

**2. GUI not showing**
```bash
# Install PyQt6
pip install PyQt6

# Linux: Install system dependencies
sudo apt-get install python3-pyqt6
```

**3. Database locked**
```python
# Enable WAL mode
engine = VoxelEngine(wal_mode=True)
```

**4. Out of memory**
```python
# Reduce batch size or n-gram level
config = {"batch_size": 5000, "max_ngram_level": 3}
```

## Configuration Files

### Training Config (JSON)

```json
{
  "batch_size": 10000,
  "validation_split": 0.1,
  "max_ngram_level": 5,
  "learning_rate": 0.01,
  "momentum_weight": 0.5,
  "temperature": 0.85,
  "save_interval": 10
}
```

### RAG Config

```json
{
  "chunk_size": 500,
  "overlap": 50,
  "top_k": 3,
  "min_score": 0.5,
  "embedding_model": "all-MiniLM-L6-v2"
}
```

## Logging

Configure logging in `voxellm/core/logger.py`:

```python
import logging

# Set log level
logging.basicConfig(level=logging.INFO)

# Training logs
logger = logging.getLogger('voxellm.training')
logger.setLevel(logging.DEBUG)

# Generation logs
logger = logging.getLogger('voxellm.generation')
logger.setLevel(logging.INFO)
```

## License

Same as parent project (MIT)

## Author

Stanley Wasonga  
BSc Cloud Computing  
stanlleylocke@gmail.com

---

**For full project documentation, see [main README](../README.md)**
