#!/usr/bin/env python3
"""Example usage of Rosetta Transformer utility modules.

This file demonstrates how to use the configuration, logging, and
reproducibility utilities.
"""

from pathlib import Path

from rosetta.utils import (
    Config,
    get_device,
    get_logger,
    print_device_info,
    set_seed,
    setup_logging,
)


def example_logging():
    """Demonstrate logging setup and usage."""
    print("\n=== Logging Example ===\n")

    # Setup logging
    setup_logging(log_file="example.log", level="INFO", detailed=False)

    # Get logger
    logger = get_logger(__name__)

    # Log messages
    logger.info("This is an info message")
    logger.debug("This debug message won't show (level is INFO)")
    logger.warning("This is a warning")
    logger.error("This is an error message")


def example_reproducibility():
    """Demonstrate reproducibility utilities."""
    print("\n=== Reproducibility Example ===\n")

    # Set random seed
    set_seed(42)
    print("Random seed set to 42")

    # Get device information
    print("\nDevice Information:")
    print_device_info()

    # Get device
    device = get_device("auto")
    print(f"\nSelected device: {device}")


def example_config_yaml():
    """Demonstrate loading config from YAML."""
    print("\n=== Config from YAML Example ===\n")

    config_path = Path("config.example.yaml")

    if config_path.exists():
        # Load config from YAML
        config = Config.load_from_yaml(config_path)

        print(f"Model name: {config.model.model_name}")
        print(f"Batch size: {config.model.batch_size}")
        print(f"Learning rate: {config.model.learning_rate}")
        print(f"Languages: {config.data.languages}")
        print(f"API port: {config.api.port}")
        print(f"Random seed: {config.seed}")

        # Create output directories
        config.create_output_dirs()
        print("\nCreated output directories:")
        print(f"  - {config.output_dir}")
        print(f"  - {config.model.checkpoint_dir}")
        print(f"  - {config.data.cache_dir}")
    else:
        print(f"Config file not found: {config_path}")


def example_config_programmatic():
    """Demonstrate creating config programmatically."""
    print("\n=== Programmatic Config Example ===\n")

    # Create config with custom values
    config = Config(seed=123, device="cpu", log_level="DEBUG")

    # Modify model config
    config.model.batch_size = 16
    config.model.learning_rate = 1e-4

    # Modify data config
    config.data.source_lang = "en"
    config.data.target_lang = "la"  # Latin

    print(f"Seed: {config.seed}")
    print(f"Batch size: {config.model.batch_size}")
    print(f"Learning rate: {config.model.learning_rate}")
    print(f"Translation: {config.data.source_lang} -> {config.data.target_lang}")

    # Save to YAML
    output_path = Path("config.generated.yaml")
    config.save_to_yaml(output_path)
    print(f"\nConfig saved to: {output_path}")


def example_config_env():
    """Demonstrate loading config from environment variables."""
    print("\n=== Config from Environment Example ===\n")

    import os

    # Set environment variables (in practice, these would be in .env file)
    os.environ["MODEL_BATCH_SIZE"] = "64"
    os.environ["MODEL_LEARNING_RATE"] = "3e-5"
    os.environ["DATA_SOURCE_LANG"] = "en"
    os.environ["DATA_TARGET_LANG"] = "grc"
    os.environ["API_PORT"] = "9000"

    # Create config (will read from env vars)
    config = Config()

    print(f"Batch size (from env): {config.model.batch_size}")
    print(f"Learning rate (from env): {config.model.learning_rate}")
    print(f"Source language (from env): {config.data.source_lang}")
    print(f"Target language (from env): {config.data.target_lang}")
    print(f"API port (from env): {config.api.port}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Rosetta Transformer Utilities - Usage Examples")
    print("=" * 60)

    example_logging()
    example_reproducibility()
    example_config_yaml()
    example_config_programmatic()
    example_config_env()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
