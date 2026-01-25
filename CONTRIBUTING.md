# Contributing to VoxelLM

Thank you for your interest in contributing to VoxelLM! This document provides guidelines for contributions.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect differing viewpoints

## How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Use issue templates** if provided
3. **Include details**:
   - VoxelLM version
   - Operating system
   - Python version
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages/logs

### Suggesting Features

1. **Search existing issues** first
2. **Describe the use case** clearly
3. **Explain the expected behavior**
4. **Consider implementation** if possible

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Write/update tests**
5. **Update documentation**
6. **Commit with clear messages**: `git commit -m 'Add amazing feature'`
7. **Push to your fork**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/voxellm.git
cd voxellm

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/voxellm.git

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install web dependencies
cd web && npm install && cd ..
```

## Code Style

### Python

- Follow **PEP 8**
- Use **type hints** where possible
- Write **docstrings** for functions/classes
- Keep functions focused and small
- Use meaningful variable names

```python
def calculate_voxel_density(
    trigram: str,
    voxel_grid: np.ndarray
) -> int:
    """Calculate density for a given trigram.
    
    Args:
        trigram: 3-character sequence
        voxel_grid: 3D numpy array
        
    Returns:
        Density count as integer
    """
    x, y, z = get_coords(trigram)
    return int(voxel_grid[x, y, z])
```

### TypeScript

- Use **TypeScript strict mode**
- Define **interfaces** for data structures
- Avoid `any` types
- Use **functional components** with hooks
- Handle errors properly

```typescript
interface TrainingConfig {
  batch_size: number;
  validation_split: number;
  max_ngram_level: number;
}

async function startTraining(
  config: TrainingConfig
): Promise<void> {
  try {
    await apiClient.startTraining(config);
  } catch (error) {
    console.error('Training failed:', error);
    throw error;
  }
}
```

## Testing

### Python Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=voxellm tests/

# Run specific test
pytest tests/test_voxel_engine.py::test_coordinate_mapping
```

### Writing Tests

```python
import pytest
from voxellm.voxel_engine import VoxelEngine

def test_coordinate_mapping():
    """Test ASCII coordinate mapping."""
    engine = VoxelEngine()
    coords = engine.get_coords("ABC")
    assert coords == (65, 66, 67)

def test_training():
    """Test basic training functionality."""
    engine = VoxelEngine()
    engine.train("Hello World")
    assert engine.voxel_grid.sum() > 0
```

## Documentation

- Update **README.md** for  new features
- Update **API.md** for API changes
- Add **docstrings** to new functions
- Update **WHITEPAPER.md** for algorithmic changes
- Include **code examples** where helpful

## Commit Messages

Use clear, descriptive commit messages:

```
Add momentum-based heuristic for A* pathfinding

- Implemented vector calculation for trajectory momentum
- Added bias towards continuing established paths
- Improved prediction accuracy on sequential data
- Closes #42
```

Format:
- First line: Summary (50 chars max)
- Blank line
- Detailed description (72 chars per line)
- Reference issues: `Closes #42`, `Fixes #38`

## Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/updates

Examples:
- `feature/trajectory-tunneling`
- `fix/coordinate-overflow`
- `docs/api-endpoints`

## Review Process

1. **Automated checks** must pass (tests, linting)
2. **Code review** by maintainers
3. **Documentation** reviewed
4. **Approval** required before merge
5. **Squash and merge** for clean history

## Areas for Contribution

### High Priority

- 🐛 Bug fixes
- 📝 Documentation improvements
- ✅ Test coverage
- 🚀 Performance optimizations

### Medium Priority

- 🎨 UI/UX improvements
- 🔧 New configuration options
- 📊 Visualization tools
- 🌐 Internationalization

### Advanced

- 🧠 Algorithm improvements
- 🔬 Research features
- 🏗️ Architecture enhancements
- 📦 Deployment automation

## Questions?

- **GitHub Discussions**: For general questions
- **Issues**: For bugs and feature requests
- **Email**: stanlleylocke@gmail.com for private inquiries

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to VoxelLM!** 🎉
