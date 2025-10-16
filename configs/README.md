# Configuration Files

This directory contains configuration files for the Rosetta Transformer project.

## Files

- **config.example.yaml**: Example configuration file with all available options

## Usage

### Create Your Configuration

1. Copy the example config:
   ```bash
   cp configs/config.example.yaml configs/config.yaml
   ```

2. Edit `configs/config.yaml` with your settings

3. Load it in your code:
   ```python
   from rosetta.utils import Config

   config = Config.load_from_yaml("configs/config.yaml")
   ```

## Configuration Sections

The configuration file includes:

- **Model Settings**: Architecture, training hyperparameters, checkpointing
- **Data Settings**: Paths, languages, data splits, preprocessing
- **API Settings**: Server configuration, CORS, serving parameters
- **MLflow Settings**: Experiment tracking configuration
- **Global Settings**: Random seed, device, logging level, output directory

## Environment Variables

You can also configure using environment variables with these prefixes:
- `MODEL_*` for model settings
- `DATA_*` for data settings
- `API_*` for API settings
- `MLFLOW_*` for MLflow settings

Example:
```bash
export MODEL_BATCH_SIZE=32
export DATA_SOURCE_LANG=en
export DATA_TARGET_LANG=grc
```

## Documentation

For detailed configuration documentation, see:
- [Utilities Documentation](../docs/reference/UTILS_README.md)
- [Quick Reference](../docs/reference/QUICK_REFERENCE.md)
