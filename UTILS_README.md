# Rosetta Transformer - Utilities Module Documentation

This document provides an overview of the utilities module implemented for the Rosetta Transformer project.

## Overview

The utilities module (`rosetta/utils/`) provides essential infrastructure for:
- **Configuration Management**: Pydantic-based settings with YAML and environment variable support
- **Logging**: Consistent logging setup across the application
- **Reproducibility**: Random seed management and device configuration

## Module Structure

```
rosetta/utils/
├── __init__.py           # Package exports
├── config.py             # Configuration management (317 lines)
├── logging.py            # Logging utilities (244 lines)
└── reproducibility.py    # Reproducibility utilities (253 lines)
```

## 1. Configuration Management (`config.py`)

### Features

- **Pydantic V2 Settings**: Type-safe configuration with validation
- **Multiple Sources**: Load from YAML files, environment variables, or programmatic defaults
- **Nested Configuration**: Organized into logical sections (Model, Data, API, MLflow)
- **Validation**: Automatic validation of configuration values with helpful error messages

### Configuration Classes

#### `ModelConfig`
Model architecture and training hyperparameters:
- Model architecture: `model_name`, `d_model`, `n_heads`, `n_layers`, `d_ff`, `dropout`
- Training: `learning_rate`, `weight_decay`, `batch_size`, `max_steps`, etc.
- Checkpointing: `checkpoint_dir`, `checkpoint_every`, `resume_from_checkpoint`

#### `DataConfig`
Data processing and loading settings:
- Paths: `data_dir`, `train_file`, `val_file`, `test_file`
- Languages: `source_lang`, `target_lang`, `languages` list
- Splits: `train_split`, `val_split`, `test_split` (validated to sum to 1.0)
- Processing: `max_source_length`, `num_workers`, `cache_dir`

#### `APIConfig`
API server configuration:
- Server: `host`, `port`, `workers`, `reload`
- CORS: `cors_origins`, `cors_credentials`, `cors_methods`, `cors_headers`
- Serving: `model_path`, `max_batch_size`, `timeout`

#### `MLflowConfig`
MLflow experiment tracking:
- `tracking_uri`, `experiment_name`, `run_name`
- `log_model`, `log_every`, `artifact_location`

#### `Config` (Main Configuration)
Combines all configuration sections and provides:
- Global settings: `seed`, `device`, `log_level`, `output_dir`
- Methods: `load_from_yaml()`, `save_to_yaml()`, `create_output_dirs()`

### Usage Examples

```python
from rosetta.utils import Config

# Load from YAML file
config = Config.load_from_yaml("config.yaml")

# Load from environment variables (with MODEL_, DATA_, API_ prefixes)
config = Config()

# Create programmatically
config = Config(
    seed=42,
    device="cuda",
    model={"batch_size": 32, "learning_rate": 5e-5},
    data={"source_lang": "en", "target_lang": "grc"}
)

# Save to YAML
config.save_to_yaml("config.yaml")

# Create output directories
config.create_output_dirs()

# Convert to dict
config_dict = config.to_dict()
```

### Environment Variables

Configuration supports environment variables with prefixes:
- `MODEL_*` for ModelConfig (e.g., `MODEL_BATCH_SIZE=32`)
- `DATA_*` for DataConfig (e.g., `DATA_SOURCE_LANG=en`)
- `API_*` for APIConfig (e.g., `API_PORT=8000`)
- `MLFLOW_*` for MLflowConfig (e.g., `MLFLOW_EXPERIMENT_NAME=experiment-1`)

## 2. Logging Utilities (`logging.py`)

### Features

- **Consistent Formatting**: Standardized log formats across the application
- **Multiple Outputs**: Log to console and/or file simultaneously
- **Level Control**: Configurable logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Third-party Suppression**: Automatically suppresses verbose third-party loggers
- **TQDM Integration**: Special handler for compatibility with progress bars

### Functions

#### `setup_logging(log_file, level, format_string, detailed)`
Configure the root logger with consistent formatting.

#### `get_logger(name)`
Get a logger instance (typically called with `__name__`).

#### `add_file_handler(logger, log_file, level, format_string)`
Add an additional file handler to a specific logger.

#### `log_dict(logger, data, level)`
Log a dictionary in a formatted, readable way.

### Classes

#### `TqdmLoggingHandler`
Custom logging handler that works with tqdm progress bars.

### Usage Examples

```python
from rosetta.utils import setup_logging, get_logger, log_dict

# Setup logging
setup_logging(
    log_file="training.log",
    level="INFO",
    detailed=True  # Include file/line info
)

# Get logger
logger = get_logger(__name__)

# Log messages
logger.info("Training started")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")

# Log dictionary
config = {"batch_size": 32, "learning_rate": 5e-5}
log_dict(logger, config, "INFO")

# Add file handler to specific logger
from rosetta.utils import add_file_handler
add_file_handler(logger, "model_training.log", "DEBUG")
```

### Log Formats

**Default Format:**
```
2025-10-16 12:30:45 - rosetta.models.transformer - INFO - Training started
```

**Detailed Format:**
```
2025-10-16 12:30:45 - rosetta.models.transformer - INFO - [train.py:42 - train_model()] - Training started
```

## 3. Reproducibility Utilities (`reproducibility.py`)

### Features

- **Random Seed Management**: Set seeds for Python, NumPy, and PyTorch
- **Device Management**: Automatic device selection and validation
- **Device Information**: Query and display CUDA/CPU/MPS information
- **Memory Tracking**: Monitor GPU memory usage
- **Cache Management**: Clear CUDA cache when needed

### Functions

#### `set_seed(seed)`
Set random seeds for all libraries (Python, NumPy, PyTorch, CUDA).

#### `get_device(device)`
Get torch device with automatic selection or validation.
- Supports: `'cuda'`, `'cpu'`, `'mps'` (Apple Silicon), `'auto'`

#### `get_device_info()`
Get detailed information about available devices as a dictionary.

#### `print_device_info()`
Print formatted device information to console.

#### `get_memory_info(device)`
Get memory statistics for the specified device (CUDA only).

#### `clear_cuda_cache()`
Clear the CUDA memory cache.

### Usage Examples

```python
from rosetta.utils import (
    set_seed,
    get_device,
    print_device_info,
    get_memory_info,
    clear_cuda_cache
)

# Set random seed for reproducibility
set_seed(42)

# Get device (automatic selection)
device = get_device("auto")
print(f"Using device: {device}")

# Print device information
print_device_info()
# Output:
# Device Information:
# ==================================================
# PyTorch Version: 2.0.0
# CUDA Available: True
# CUDA Version: 11.8
# CUDA Device Count: 1
# CUDA Devices:
#   [0] NVIDIA GeForce RTX 3090
#       Total Memory: 24.00 GB
#       Compute Capability: 8.6
# ==================================================

# Get memory info
mem_info = get_memory_info(device)
print(f"Allocated: {mem_info['allocated']:.2f} GB")
print(f"Free: {mem_info['free']:.2f} GB")

# Clear CUDA cache
clear_cuda_cache()
```

## Package Exports

All main classes and functions are exported from `rosetta.utils`:

```python
from rosetta.utils import (
    # Config classes
    Config,
    ModelConfig,
    DataConfig,
    APIConfig,
    MLflowConfig,

    # Logging
    setup_logging,
    get_logger,
    add_file_handler,
    log_dict,
    TqdmLoggingHandler,
    DEFAULT_FORMAT,
    DETAILED_FORMAT,

    # Reproducibility
    set_seed,
    get_device,
    get_device_info,
    print_device_info,
    get_memory_info,
    clear_cuda_cache,
)
```

## Example Configuration File

See `config.example.yaml` for a complete example configuration file.

## Integration Example

Typical usage in a training script:

```python
from rosetta.utils import (
    Config,
    setup_logging,
    get_logger,
    set_seed,
    get_device,
    print_device_info
)

def main():
    # Load configuration
    config = Config.load_from_yaml("config.yaml")

    # Setup logging
    setup_logging(
        log_file=f"{config.output_dir}/training.log",
        level=config.log_level,
        detailed=True
    )
    logger = get_logger(__name__)

    # Set random seed
    set_seed(config.seed)
    logger.info(f"Random seed set to {config.seed}")

    # Get device
    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    # Print device info
    print_device_info()

    # Create output directories
    config.create_output_dirs()
    logger.info("Output directories created")

    # Your training code here...
    logger.info("Starting training...")

if __name__ == "__main__":
    main()
```

## Testing

Run the example script to see all utilities in action:

```bash
python examples_utils_usage.py
```

## Design Principles

1. **Type Safety**: Full type hints and Pydantic validation
2. **Documentation**: Comprehensive docstrings with examples
3. **Flexibility**: Multiple configuration sources (YAML, env, programmatic)
4. **Production Ready**: Error handling, validation, and logging best practices
5. **Clean API**: Intuitive imports and clear function signatures
