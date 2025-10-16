# Rosetta-Transformer-for-Antiquity
[![CI](https://github.com/AyhamJo7/Rosetta-Transformer-for-Antiquity/actions/workflows/ci.yml/badge.svg)](https://github.com/AyhamJo7/Rosetta-Transformer-for-Antiquity/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="rosetta/Rosetta-Transformer-for-Antiquity.png" alt="Rosetta-Transformer-for-Antiquity Logo" width="400"/>
</p>

## Overview

**Rosetta Transformer for Antiquity** is a state-of-the-art multilingual NLP system designed to revolutionize the study of ancient and low-resource languages. Built on modern Transformer architectures, this project provides researchers, linguists, and historians with powerful tools for analyzing, translating, and understanding ancient texts.

The system bridges the gap between cutting-edge machine learning and classical scholarship, offering a comprehensive platform for:

- **Ancient Language Processing**: Support for Ancient Greek (grc), Latin (la), Classical Arabic (ar), and other historical languages
- **Machine Translation**: Neural translation between ancient and modern languages
- **Linguistic Analysis**: NER, POS tagging, morphological analysis, and relation extraction
- **Interactive Research**: Web-based tools for corpus exploration and model-assisted annotation
- **Reproducible Science**: Complete experiment tracking and configuration management

### Key Features

- **Production-Ready Architecture**: FastAPI-based REST API with comprehensive error handling
- **Flexible Model Training**: Support for pretraining, fine-tuning, and transfer learning
- **Expert-in-the-Loop**: Human feedback integration for continuous model improvement
- **Scalable Infrastructure**: Docker support for both CPU and GPU deployment
- **Comprehensive Evaluation**: Statistical metrics with bootstrap confidence intervals
- **Configuration Management**: YAML-based configs with environment variable support

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
  - [Configuration](#configuration)
  - [Training Models](#training-models)
  - [Running the API Server](#running-the-api-server)
  - [Using the CLI](#using-the-cli)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git
- (Optional) CUDA-compatible GPU for training
- (Optional) Docker for containerized deployment

### Step 1: Clone the Repository

```bash
git clone https://github.com/AyhamJo7/Rosetta-Transformer-for-Antiquity.git
cd Rosetta-Transformer-for-Antiquity
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.in

# Install development dependencies (for contributors)
pip install -r requirements-dev.in

# Install pre-commit hooks (optional, recommended for development)
pre-commit install
```

### Step 4: Verify Installation

```bash
# Test imports
python -c "from rosetta.utils import Config, setup_logging, set_seed; print('Installation successful!')"

# Run example script
python examples/examples_utils_usage.py
```

---

## Quick Start

### 1. Configure Your Project

```bash
# Copy example configuration
cp configs/config.example.yaml configs/config.yaml

# Edit configuration with your settings
# Adjust paths, model parameters, and language settings
nano configs/config.yaml
```

### 2. Run the API Server

```bash
# Start the development server
uvicorn rosetta.api.main:app --reload --host 0.0.0.0 --port 8000

# The API will be available at http://localhost:8000
# Interactive documentation at http://localhost:8000/docs
```

### 3. Use the CLI

```bash
# Process ancient texts
python -m rosetta.cli process --input data/raw --output data/processed

# Train a model
python -m rosetta.cli train --config configs/config.yaml

# Evaluate model performance
python -m rosetta.cli evaluate --model-path checkpoints/model.pt --test-data data/test
```

---

## Project Structure

```
Rosetta-Transformer-for-Antiquity/
│
├── rosetta/                      # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # Command-line interface
│   │
│   ├── api/                     # FastAPI REST API
│   │   ├── __init__.py
│   │   ├── main.py              # API entry point and router configuration
│   │   ├── inference.py         # Model inference engine with caching
│   │   ├── feedback.py          # Human-in-the-loop feedback system
│   │   └── search.py            # Semantic search and corpus exploration
│   │
│   ├── models/                  # Neural network architectures
│   │   ├── __init__.py
│   │   ├── base.py              # Base model classes and interfaces
│   │   ├── pretraining.py       # Domain adaptation and pretraining
│   │   ├── seq2seq.py           # Sequence-to-sequence models (translation)
│   │   └── token_tasks.py       # Token classification (NER, POS tagging)
│   │
│   ├── data/                    # Data processing and management
│   │   ├── __init__.py
│   │   ├── corpus.py            # Corpus loading and management
│   │   ├── cleaning.py          # Text normalization and preprocessing
│   │   ├── alignment.py         # Parallel text alignment
│   │   ├── annotation.py        # Annotation utilities and formats
│   │   └── schemas.py           # Data validation schemas (Pydantic)
│   │
│   ├── evaluation/              # Model evaluation and metrics
│   │   ├── __init__.py
│   │   ├── metrics.py           # BLEU, exact match, bootstrap CI
│   │   └── validators.py        # Output validation and quality checks
│   │
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── config.py            # Configuration management (Pydantic)
│       ├── logging.py           # Logging utilities
│       └── reproducibility.py   # Random seed and device management
│
├── configs/                     # Configuration files
│   ├── README.md
│   └── config.example.yaml      # Example configuration template
│
├── docker/                      # Docker deployment
│   ├── README.md
│   ├── Dockerfile.cpu           # CPU-only Docker image
│   └── Dockerfile.cuda          # GPU-accelerated Docker image
│
├── examples/                    # Usage examples
│   ├── README.md
│   └── examples_utils_usage.py  # Utility modules demonstration
│
├── scripts/                     # Utility scripts
│   └── fixes/                   # Type checking and code quality fixes
│       ├── README.md
│       └── *.py                 # Mypy error fix scripts
│
├── docs/                        # Documentation
│   ├── reference/               # API and module references
│   │   ├── UTILS_README.md      # Utilities documentation
│   │   └── QUICK_REFERENCE.md   # Quick reference guide
│   └── implementation/          # Implementation notes
│       ├── IMPLEMENTATION_SUMMARY.md
│       └── MYPY_FIXES_SUMMARY.md
│
├── tests/                       # Test suite
│   └── __init__.py
│
├── .github/                     # GitHub configuration
│   ├── workflows/
│   │   └── ci.yml               # CI/CD pipeline (type checking, tests)
│   └── ISSUE_TEMPLATE/          # Issue templates
│
├── requirements.in              # Core dependencies
├── requirements-dev.in          # Development dependencies
├── pyproject.toml               # Project metadata and tool configuration
├── .pre-commit-config.yaml      # Pre-commit hooks (black, ruff, mypy)
└── README.md                    # This file
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| **rosetta.api** | REST API for model serving, inference, and user interaction |
| **rosetta.models** | Neural network implementations for various NLP tasks |
| **rosetta.data** | Data loading, cleaning, and preprocessing pipelines |
| **rosetta.evaluation** | Metrics computation and model validation |
| **rosetta.utils** | Configuration, logging, and reproducibility utilities |
| **rosetta.cli** | Command-line interface for training and evaluation |

---

## Usage Guide

### Configuration

The project uses a flexible configuration system supporting YAML files, environment variables, and programmatic setup.

#### Create Your Configuration

```bash
# Copy the example config
cp configs/config.example.yaml configs/config.yaml

# Edit with your preferred editor
nano configs/config.yaml
```

#### Configuration Sections

**Model Settings**
```yaml
model:
  model_name: "Helsinki-NLP/opus-mt-en-grc"  # Base model
  d_model: 512                                # Model dimension
  n_heads: 8                                  # Attention heads
  n_layers: 6                                 # Transformer layers
  learning_rate: 0.0001
  batch_size: 32
  checkpoint_dir: "checkpoints"
```

**Data Settings**
```yaml
data:
  data_dir: "data"
  source_lang: "en"                           # Source language code
  target_lang: "grc"                          # Ancient Greek
  languages: ["en", "grc", "la"]              # Supported languages
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

**API Settings**
```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  model_path: "checkpoints/best_model.pt"
  max_batch_size: 64
```

#### Environment Variables

Override settings using environment variables:

```bash
export MODEL_BATCH_SIZE=64
export DATA_SOURCE_LANG=en
export DATA_TARGET_LANG=grc
export API_PORT=8080
```

#### Programmatic Configuration

```python
from rosetta.utils import Config, ModelConfig, DataConfig

# Create configuration programmatically
config = Config(
    model=ModelConfig(
        model_name="Helsinki-NLP/opus-mt-en-grc",
        batch_size=32,
        learning_rate=1e-4
    ),
    data=DataConfig(
        source_lang="en",
        target_lang="grc",
        data_dir="data"
    ),
    seed=42,
    device="cuda"
)

# Save configuration
config.save_to_yaml("configs/my_config.yaml")
```

### Training Models

#### Basic Training

```python
from rosetta.utils import Config, setup_logging, set_seed, get_device
from rosetta.models import Seq2SeqModel
from rosetta.data import load_corpus

# Load configuration
config = Config.load_from_yaml("configs/config.yaml")

# Setup environment
setup_logging(level=config.log_level)
set_seed(config.seed)
device = get_device(config.device)

# Load data
train_data, val_data = load_corpus(config.data)

# Initialize model
model = Seq2SeqModel(config.model).to(device)

# Train
model.fit(train_data, val_data, config)
```

#### Using the CLI

```bash
# Train from configuration
python -m rosetta.cli train --config configs/config.yaml

# Resume from checkpoint
python -m rosetta.cli train --config configs/config.yaml --resume checkpoints/last.pt

# Fine-tune pretrained model
python -m rosetta.cli finetune \
    --base-model checkpoints/pretrained.pt \
    --config configs/finetune.yaml \
    --task translation
```

### Running the API Server

#### Development Mode

```bash
# Start with auto-reload
uvicorn rosetta.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode

```bash
# Start with multiple workers
uvicorn rosetta.api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

#### Docker Deployment

```bash
# Build CPU image
docker build -f docker/Dockerfile.cpu -t rosetta:cpu .

# Build GPU image
docker build -f docker/Dockerfile.cuda -t rosetta:cuda .

# Run CPU container
docker run -p 8000:8000 -v $(pwd)/configs:/app/configs rosetta:cpu

# Run GPU container
docker run --gpus all -p 8000:8000 -v $(pwd)/configs:/app/configs rosetta:cuda
```

### Using the CLI

The CLI provides commands for common workflows:

#### Process Text Data

```bash
# Clean and normalize corpus
python -m rosetta.cli process \
    --input data/raw/corpus.txt \
    --output data/processed/corpus.txt \
    --language grc \
    --normalize

# Batch process directory
python -m rosetta.cli process \
    --input data/raw/ \
    --output data/processed/ \
    --language grc \
    --workers 4
```

#### Evaluate Models

```bash
# Evaluate on test set
python -m rosetta.cli evaluate \
    --model checkpoints/best_model.pt \
    --test-data data/test.json \
    --metrics bleu,exact_match \
    --output results/evaluation.json

# Evaluate with bootstrap confidence intervals
python -m rosetta.cli evaluate \
    --model checkpoints/best_model.pt \
    --test-data data/test.json \
    --bootstrap-samples 1000 \
    --confidence 0.95
```

---

## API Documentation

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example API Calls

#### Translation Endpoint

```bash
# Translate English to Ancient Greek
curl -X POST "http://localhost:8000/api/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "source_lang": "en",
    "target_lang": "grc"
  }'
```

**Response:**
```json
{
  "translation": "Χαῖρε, κόσμε!",
  "confidence": 0.92,
  "metadata": {
    "model": "opus-mt-en-grc",
    "processing_time_ms": 145
  }
}
```

#### Batch Translation

```bash
curl -X POST "http://localhost:8000/api/v1/translate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello", "Goodbye", "Thank you"],
    "source_lang": "en",
    "target_lang": "grc"
  }'
```

#### Named Entity Recognition

```bash
curl -X POST "http://localhost:8000/api/v1/ner" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Socrates was a philosopher in Athens.",
    "language": "en"
  }'
```

#### Feedback Submission

```bash
# Submit correction for model improvement
curl -X POST "http://localhost:8000/api/v1/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "trans_12345",
    "correction": "Improved translation here",
    "rating": 4
  }'
```

### Python Client Example

```python
import requests

class RosettaClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def translate(self, text, source_lang="en", target_lang="grc"):
        response = requests.post(
            f"{self.base_url}/api/v1/translate",
            json={
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang
            }
        )
        return response.json()

# Usage
client = RosettaClient()
result = client.translate("Philosophy is the love of wisdom")
print(result["translation"])
```

---

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.in

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Type checking
mypy rosetta/

# Code formatting
black rosetta/
ruff check rosetta/
```

### Code Quality Standards

This project enforces:
- **Type Hints**: Full mypy compliance (Python 3.11+)
- **Code Formatting**: Black and Ruff
- **Testing**: Pytest with coverage reporting
- **Documentation**: Comprehensive docstrings
- **Pre-commit Hooks**: Automated quality checks

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rosetta --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v -s
```

### Continuous Integration

The project uses GitHub Actions for CI/CD:
- Type checking with mypy
- Code formatting validation
- Unit and integration tests
- Documentation building

All pull requests must pass CI checks before merging.

---

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Ensure all tests pass**: `pytest`
5. **Run code quality checks**: `pre-commit run --all-files`
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed
- Keep commits atomic and well-described

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{rosetta_transformer_2025,
  title={Rosetta Transformer for Antiquity: Neural Machine Translation for Ancient Languages},
  author={Your Name},
  year={2025},
  url={https://github.com/AyhamJo7/Rosetta-Transformer-for-Antiquity}
}
```

---

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [HuggingFace Transformers](https://huggingface.co/transformers/)
- API powered by [FastAPI](https://fastapi.tiangolo.com/)
- Inspired by the [Rosetta Stone](https://en.wikipedia.org/wiki/Rosetta_Stone) and its role in deciphering ancient languages

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/AyhamJo7/Rosetta-Transformer-for-Antiquity/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AyhamJo7/Rosetta-Transformer-for-Antiquity/discussions)
- **Documentation**: [Project Wiki](https://github.com/AyhamJo7/Rosetta-Transformer-for-Antiquity/wiki)

---

**Made with dedication to preserving and understanding our shared human heritage through technology.**
