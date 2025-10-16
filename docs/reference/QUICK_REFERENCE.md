# Rosetta Transformer Utils - Quick Reference

## Installation

```bash
pip install -r requirements.in
```

## Basic Usage

### Configuration

```python
from rosetta.utils import Config

# From YAML
config = Config.load_from_yaml("config.yaml")

# From environment variables (MODEL_*, DATA_*, API_*, MLFLOW_*)
config = Config()

# Programmatic
config = Config(
    seed=42,
    device="cuda",
    model={"batch_size": 32, "learning_rate": 5e-5}
)

# Save to YAML
config.save_to_yaml("config.yaml")

# Create directories
config.create_output_dirs()
```

### Logging

```python
from rosetta.utils import setup_logging, get_logger

# Setup
setup_logging(log_file="train.log", level="INFO")

# Use
logger = get_logger(__name__)
logger.info("Training started")
logger.debug("Debug info")
logger.warning("Warning!")
```

### Reproducibility

```python
from rosetta.utils import set_seed, get_device, print_device_info

# Set seed for reproducibility
set_seed(42)

# Get device
device = get_device("auto")  # or "cuda", "cpu", "mps"

# Show device info
print_device_info()
```

## Complete Example

```python
from rosetta.utils import (
    Config, setup_logging, get_logger,
    set_seed, get_device, print_device_info
)

def main():
    # Load config
    config = Config.load_from_yaml("config.yaml")

    # Setup logging
    setup_logging(
        log_file=f"{config.output_dir}/training.log",
        level=config.log_level
    )
    logger = get_logger(__name__)

    # Reproducibility
    set_seed(config.seed)
    device = get_device(config.device)

    logger.info(f"Seed: {config.seed}")
    logger.info(f"Device: {device}")

    print_device_info()
    config.create_output_dirs()

    # Your code here...
    logger.info("Training started")

if __name__ == "__main__":
    main()
```

## Configuration Sections

### Model (model.*)
- model_name, d_model, n_heads, n_layers
- learning_rate, batch_size, max_steps
- checkpoint_dir, checkpoint_every

### Data (data.*)
- data_dir, train_file, val_file, test_file
- source_lang, target_lang, languages
- train_split, val_split, test_split

### API (api.*)
- host, port, workers, reload
- cors_origins, cors_credentials
- model_path, max_batch_size

### MLflow (mlflow.*)
- tracking_uri, experiment_name
- log_model, log_every

## Environment Variables

```bash
# Model
export MODEL_BATCH_SIZE=32
export MODEL_LEARNING_RATE=5e-5

# Data
export DATA_SOURCE_LANG=en
export DATA_TARGET_LANG=grc

# API
export API_PORT=8000

# MLflow
export MLFLOW_EXPERIMENT_NAME=my-experiment
```

## Common Functions

| Function | Purpose |
|----------|---------|
| `Config.load_from_yaml(path)` | Load config from YAML |
| `setup_logging(log_file, level)` | Setup logging |
| `get_logger(name)` | Get logger instance |
| `set_seed(seed)` | Set random seeds |
| `get_device(device)` | Get torch device |
| `print_device_info()` | Print device info |
| `get_memory_info(device)` | Get GPU memory |
| `clear_cuda_cache()` | Clear CUDA cache |

## File Locations

- Config classes: `rosetta/utils/config.py`
- Logging: `rosetta/utils/logging.py`
- Reproducibility: `rosetta/utils/reproducibility.py`
- Example config: `configs/config.example.yaml`
- Usage examples: `examples/examples_utils_usage.py`
- Full docs: `docs/reference/UTILS_README.md`
