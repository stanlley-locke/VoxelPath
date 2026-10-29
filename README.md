# VoxelLM - Geometric 3D Coordinate-Mapped Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)

A revolutionary **Geometric Encoding** language model that processes text by mapping character sequences into 3D spatial coordinates, achieving transparent and efficient pattern recognition with minimal memory footprint.

## 🌟 Overview

VoxelLM implements the **VoxelPath** paradigm - a deterministic spatial AI system that treats language as geometric trajectories through a structured 3D coordinate system, rather than probabilistic token distributions.

### Key Features

- ** Geometric Encoding**: Maps 3-character sequences (trigrams) to unique (x,y,z) coordinates in a 128³ voxel cube
- ** Ultra-Lightweight**: ~8.4 MB memory footprint (entire model fits in CPU L3 cache)
- ** O(n) Complexity**: Linear training and inference time
- ** Transparent**: Visualizable decision paths, no "black box"
- ** Hybrid Architecture**: Neural n-gram scorer for enhanced accuracy
- ** RAG Integration**: Document retrieval with embedding-based search
- ** Modern Web UI**: TypeScript interface with real-time training visualization

## 📖 Whitepaper

VoxelLM is based on the **VoxelPath** research by Stanley Wasonga (BSc Cloud Computing).

**Core Concept**: Instead of abstract semantic vectors, VoxelPath utilizes the intrinsic numerical properties of ASCII characters to create a coordinate base where every trigram occupies a unique physical point in 3D space. Learning becomes spatial density mapping as data "carves paths" through this space.

[Read Full Whitepaper](./WHITEPAPER.md)

### 🌐 Interactive Whitepaper

The whitepaper is also available as a responsive GitHub Pages site:

**[Open the VoxelPath whitepaper site](https://stanlley-locke.github.io/VoxelPath/)**

To run it locally:

```bash
cd web
npm install
npm run dev
```

Every push to `main` builds and deploys the site through [`.github/workflows/deploy-pages.yml`](./.github/workflows/deploy-pages.yml).

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   VoxelLM System                     │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────┐    ┌──────────────┐               │
│  │  Voxel Grid  │───▶│  A* Pathfind │               │
│  │  (128³ cube) │    │   Inference   │               │
│  └──────────────┘    └──────────────┘               │
│         │                                             │
│         ▼                                             │
│  ┌──────────────┐    ┌──────────────┐               │
│  │ Neural N-gram│───▶│   Hybrid     │               │
│  │    Scorer    │    │  Generation  │               │
│  └──────────────┘    └──────────────┘               │
│                                                       │
│  ┌──────────────┐    ┌──────────────┐               │
│  │ RAG Document │───▶│   Retrieval  │               │
│  │   Embedding  │    │   Augmented  │               │
│  └──────────────┘    └──────────────┘               │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### Components

1. **Voxel Engine** (`voxellm/voxel_engine.py`)
   - 3D coordinate mapping
   - Density-based learning
   - A* pathfinding inference

2. **Neural Scorer** (`voxellm/neural_scorer.py`)
   - Momentum-weighted n-gram predictions
   - PyTorch-based probability estimation

3. **RAG System** (`voxellm/rag_system.py`)
   - Document chunking and indexing
   - Embedding-based retrieval
   - Context-enhanced generation

4. **FastAPI Server** (`voxellm/server_main.py`)
   - RESTful API endpoints
   - Real-time training status
   - Model management

5. **Web Interface** (`web/`)
   - React + TypeScript UI
   - Live training visualization
   - Chat, model management, analytics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (for web interface)
- 16GB RAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/stanlley-locke/voxellm.git
cd voxellm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install web dependencies
cd web
npm install
cd ..
```

### Running the System

**1. Start API Server**
```bash
python -m voxellm.server_main
```
Server runs at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

**2. Start Web Interface**
```bash
cd web
npm run dev
```
Web UI runs at `http://localhost:5173`

## 📚 Usage Examples

### Training a Model

```python
from voxellm.voxel_engine import VoxelEngine

# Initialize engine
engine = VoxelEngine()

# Train on corpus
corpus = "The quick brown fox jumps over the lazy dog."
engine.train(corpus)

# Generate text
result = engine.generate("The quick", max_length=20)
print(result)  # "The quick brown fox..."
```

### Using the API

```bash
# Train model
curl -X POST http://localhost:8000/training/start \
  -H "Content-Type: application/json" \
  -d '{"corpus": "Training text here..."}'

# Generate text
curl -X POST http://localhost:8000/generation/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_length": 50}'

# Chat interface
curl -X POST http://localhost:8000/generation/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### RAG Document Retrieval

```python
from voxellm.rag_system import SimpleVectorStore

# Initialize RAG
rag = SimpleVectorStore()

# Ingest documents
docs = [
    {"content": "VoxelLM uses 3D coordinates...", "source": "intro.txt"},
    {"content": "Training is O(n) complexity...", "source": "perf.txt"}
]
rag.ingest_documents(docs)

# Query with retrieval
context = rag.query("How does VoxelLM work?", top_k=2)
print(context)
```

## 🎯 Key Algorithms

### 1. Coordinate Mapping

```python
def get_coords(trigram: str) -> tuple[int, int, int]:
    """Maps 3-char sequence to (x, y, z) coordinates"""
    return tuple(ord(c) & 0x7F for c in trigram)

# Example: "The" → (84, 104, 101)
```

### 2. Voxel Training (O(n))

```python
for i in range(len(text) - 2):
    trigram = text[i:i+3]
    x, y, z = get_coords(trigram)
    voxel_grid[x, y, z] += 1  # Increment density
```

### 3. A* Pathfinding Inference

```python
def a_star_generate(current_coords, target_length):
    while len(output) < target_length:
        next_char = find_best_path(
            current_coords,
            cost_func=lambda v: 1 / (voxel_density[v] + 1),
            heuristic=momentum_based_heuristic
        )
        output += next_char
```

## 📊 Performance Metrics

| Metric | VoxelLM | Traditional LLM |
|--------|---------|----------------|
| Memory | ~8.4 MB | 2-175 GB |
| Training Complexity | O(n) | O(n²) |
| Hardware | CPU | GPU Required |
| Explainability | Transparent | Black Box |
| Edge Deployment | ✅ Yes | ❌ No |

## 🗂️ Project Structure

```
voxellm/
├── voxellm/                 # Core engine
│   ├── voxel_engine.py     # 3D coordinate system
│   ├── neural_scorer.py    # Hybrid scoring
│   ├── rag_system.py       # Document retrieval
│   ├── server_main.py      # FastAPI server
│   └── models/             # Saved models
├── web/                     # TypeScript UI
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── lib/            # API client
│   │   └── types/          # TypeScript types
│   └── package.json
├── tests/                   # Test suite
├── requirements.txt
├── WHITEPAPER.md
└── README.md
```

## 🔬 Technical Innovations

### 1. Semantic Halo
Links uppercase/lowercase variants through geometric proximity (32-offset in ASCII).

### 2. Trajectory Tunneling
Creates direct vector links for high-frequency long words, skipping intermediate calculations.

### 3. Temporal Decay
Self-cleaning memory that evaporates noise while retaining structural knowledge.

### 4. Momentum-Based Prediction
Physics-inspired heuristic that biases predictions to continue established trajectories.

## 🌐 Web Interface Features

- **Chat**: Real-time conversation with loaded models
- **Training**: Upload corpus, configure parameters, monitor progress
- **Models**: Save, load, export trained models
- **Analytics**: Live system metrics and performance stats

## 📝 API Documentation

Full API documentation available at `http://localhost:8000/docs` when server is running.

### Main Endpoints

- `POST /generation/chat` - Chat interface
- `POST /generation/generate` - Text generation
- `POST /training/start` - Start training
- `GET /training/status` - Training progress
- `GET /models/list` - List saved models
- `POST /models/save` - Save current model
- `POST /models/load/{name}` - Load model

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_voxel_engine.py

# With coverage
pytest --cov=voxellm tests/
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Stanley Wasonga**
- BSc Cloud Computing
- Email: stanlleylocke@gmail.com

## 🙏 Acknowledgments

- VoxelPath whitepaper research
- FastAPI and React communities
- Open source AI/ML ecosystem

## 📚 Citation

```bibtex
@software{voxellm2026,
  author = {Wasonga, Stanley},
  title = {VoxelLM: Geometric 3D Coordinate-Mapped Language Model},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/voxellm}
}
```

---

**Built with  using Stanlley Locke VoxelPath**
