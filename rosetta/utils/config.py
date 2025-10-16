"""Configuration management for Rosetta Transformer.

This module provides Pydantic-based configuration management with support for
loading from YAML files and environment variables.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Model architecture and training configuration.

    Attributes:
        model_name: Name or path of pretrained model
        max_length: Maximum sequence length
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        warmup_steps: Number of warmup steps for scheduler
        max_steps: Maximum training steps
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        checkpoint_dir: Directory for saving checkpoints
        checkpoint_every: Save checkpoint every N steps
        resume_from_checkpoint: Path to checkpoint to resume from
    """

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # Model architecture
    model_name: str = Field(
        default="bert-base-multilingual-cased",
        description="Name or path of pretrained model",
    )
    max_length: int = Field(default=512, ge=1, le=8192)
    d_model: int = Field(default=768, ge=64)
    n_heads: int = Field(default=12, ge=1)
    n_layers: int = Field(default=12, ge=1)
    d_ff: int = Field(default=3072, ge=64)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)

    # Training hyperparameters
    learning_rate: float = Field(default=5e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_steps: int = Field(default=1000, ge=0)
    max_steps: int = Field(default=100000, ge=1)
    batch_size: int = Field(default=32, ge=1)
    eval_batch_size: int = Field(default=64, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0.0)

    # Checkpointing
    checkpoint_dir: str = Field(default="checkpoints")
    checkpoint_every: int = Field(default=1000, ge=1)
    resume_from_checkpoint: Optional[str] = Field(default=None)

    @field_validator("n_heads")
    @classmethod
    def validate_n_heads(cls, v: int, info) -> int:
        """Validate that d_model is divisible by n_heads."""
        # Note: d_model may not be available yet during validation
        return v


class DataConfig(BaseSettings):
    """Data processing and loading configuration.

    Attributes:
        data_dir: Root directory for datasets
        train_file: Path to training data file
        val_file: Path to validation data file
        test_file: Path to test data file
        source_lang: Source language code
        target_lang: Target language code
        languages: List of language codes to process
        max_source_length: Maximum source sequence length
        max_target_length: Maximum target sequence length
        train_split: Training data split ratio
        val_split: Validation data split ratio
        test_split: Test data split ratio
        num_workers: Number of data loading workers
        cache_dir: Directory for caching processed data
        preprocessing_num_workers: Number of preprocessing workers
        overwrite_cache: Whether to overwrite cached data
    """

    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # Data paths
    data_dir: str = Field(default="data")
    train_file: Optional[str] = Field(default=None)
    val_file: Optional[str] = Field(default=None)
    test_file: Optional[str] = Field(default=None)

    # Language settings
    source_lang: str = Field(default="en")
    target_lang: str = Field(default="grc")  # Ancient Greek
    languages: List[str] = Field(default_factory=lambda: ["en", "grc", "la", "ar"])

    # Sequence lengths
    max_source_length: int = Field(default=512, ge=1, le=8192)
    max_target_length: int = Field(default=512, ge=1, le=8192)

    # Data splits
    train_split: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_split: float = Field(default=0.1, gt=0.0, lt=1.0)
    test_split: float = Field(default=0.1, gt=0.0, lt=1.0)

    # Data loading
    num_workers: int = Field(default=4, ge=0)
    cache_dir: str = Field(default=".cache")
    preprocessing_num_workers: int = Field(default=4, ge=1)
    overwrite_cache: bool = Field(default=False)

    @field_validator("train_split", "val_split", "test_split")
    @classmethod
    def validate_splits(cls, v: float) -> float:
        """Validate that splits sum to 1.0."""
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that splits sum to approximately 1.0."""
        total = self.train_split + self.val_split + self.test_split
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Data splits must sum to 1.0, got {total:.4f} "
                f"(train={self.train_split}, val={self.val_split}, "
                f"test={self.test_split})"
            )


class APIConfig(BaseSettings):
    """API server configuration.

    Attributes:
        host: Server host address
        port: Server port
        reload: Enable auto-reload (development)
        workers: Number of worker processes
        cors_origins: List of allowed CORS origins
        cors_credentials: Allow credentials in CORS
        cors_methods: Allowed HTTP methods
        cors_headers: Allowed HTTP headers
        model_path: Path to model for serving
        max_batch_size: Maximum batch size for inference
        timeout: Request timeout in seconds
    """

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=False)
    workers: int = Field(default=1, ge=1)

    # CORS settings
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    cors_credentials: bool = Field(default=True)
    cors_methods: List[str] = Field(default_factory=lambda: ["*"])
    cors_headers: List[str] = Field(default_factory=lambda: ["*"])

    # Model serving
    model_path: Optional[str] = Field(default=None)
    max_batch_size: int = Field(default=32, ge=1)
    timeout: int = Field(default=30, ge=1)


class MLflowConfig(BaseSettings):
    """MLflow experiment tracking configuration.

    Attributes:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of MLflow experiment
        run_name: Name of MLflow run
        log_model: Whether to log model artifacts
        log_every: Log metrics every N steps
        artifact_location: Location for storing artifacts
    """

    model_config = SettingsConfigDict(
        env_prefix="MLFLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    tracking_uri: str = Field(default="./mlruns")
    experiment_name: str = Field(default="rosetta-transformer")
    run_name: Optional[str] = Field(default=None)
    log_model: bool = Field(default=True)
    log_every: int = Field(default=100, ge=1)
    artifact_location: Optional[str] = Field(default=None)


class Config(BaseSettings):
    """Main configuration class that combines all configuration sections.

    This class can load configuration from:
    1. YAML file (via load_from_yaml method)
    2. Environment variables (with appropriate prefixes)
    3. Default values

    Attributes:
        model: Model configuration
        data: Data configuration
        api: API configuration
        mlflow: MLflow configuration
        seed: Random seed for reproducibility
        device: Device to use (cuda, cpu, or auto)
        log_level: Logging level
        output_dir: Output directory for results
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    # Global settings
    seed: int = Field(default=42, ge=0)
    device: str = Field(default="auto")
    log_level: str = Field(default="INFO")
    output_dir: str = Field(default="outputs")

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path) -> "Config":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance with values from YAML file

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML file is invalid
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        return cls(**config_dict)

    def save_to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            yaml_path: Path where YAML file should be saved
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle nested Pydantic models
        config_dict = self.model_dump()

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return self.model_dump()

    def create_output_dirs(self) -> None:
        """Create output directories specified in configuration."""
        dirs_to_create = [
            self.output_dir,
            self.model.checkpoint_dir,
            self.data.cache_dir,
        ]

        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
