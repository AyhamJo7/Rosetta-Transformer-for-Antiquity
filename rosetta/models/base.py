"""Base model classes for Rosetta Transformer.

This module provides abstract base classes and utilities for model training,
including base model interface, trainer with early stopping and checkpointing,
and configuration dataclasses.
"""

import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, get_scheduler

from rosetta.utils.logging import get_logger
from rosetta.utils.reproducibility import set_seed

logger = get_logger(__name__)


@dataclass
class ModelOutput:
    """Container for model outputs.

    Attributes:
        loss: Scalar loss value (optional)
        logits: Model predictions/logits
        hidden_states: Hidden states from intermediate layers (optional)
        attentions: Attention weights (optional)
        metrics: Dictionary of additional metrics
    """

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary, moving tensors to CPU."""
        result = {}

        if self.loss is not None:
            result["loss"] = self.loss.detach().cpu().item()
        if self.logits is not None:
            result["logits"] = self.logits.detach().cpu()
        if self.hidden_states is not None:
            result["hidden_states"] = tuple(
                h.detach().cpu() for h in self.hidden_states
            )
        if self.attentions is not None:
            result["attentions"] = tuple(a.detach().cpu() for a in self.attentions)
        result["metrics"] = self.metrics

        return result


@dataclass
class TrainingArguments:
    """Arguments for model training.

    Attributes:
        output_dir: Directory for outputs (checkpoints, logs)
        num_train_epochs: Total number of training epochs
        max_steps: Maximum number of training steps (overrides num_train_epochs)
        per_device_train_batch_size: Batch size per device during training
        per_device_eval_batch_size: Batch size per device during evaluation
        gradient_accumulation_steps: Number of updates steps to accumulate gradients
        learning_rate: Initial learning rate
        weight_decay: Weight decay coefficient
        adam_beta1: Beta1 for Adam optimizer
        adam_beta2: Beta2 for Adam optimizer
        adam_epsilon: Epsilon for Adam optimizer
        max_grad_norm: Maximum gradient norm for clipping
        warmup_steps: Number of warmup steps for learning rate scheduler
        warmup_ratio: Ratio of warmup steps to total steps (overrides warmup_steps)
        lr_scheduler_type: Type of learning rate scheduler
        logging_steps: Log every X steps
        eval_steps: Evaluate every X steps
        save_steps: Save checkpoint every X steps
        save_total_limit: Maximum number of checkpoints to keep
        fp16: Whether to use mixed precision training (FP16)
        bf16: Whether to use bfloat16 precision
        gradient_checkpointing: Whether to use gradient checkpointing
        dataloader_num_workers: Number of subprocesses for data loading
        seed: Random seed for reproducibility
        local_rank: Local rank for distributed training
        ddp_find_unused_parameters: Whether to find unused parameters in DDP
        early_stopping_patience: Number of evaluations with no improvement before stopping
        early_stopping_threshold: Minimum change to qualify as improvement
        load_best_model_at_end: Whether to load best model at end of training
        metric_for_best_model: Metric to use for selecting best model
        greater_is_better: Whether higher metric value is better
        resume_from_checkpoint: Path to checkpoint to resume from
        mlflow_tracking_uri: MLflow tracking URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_run_name: MLflow run name
        log_model_every_n_steps: Log model to MLflow every N steps (0 to disable)
    """

    # Output
    output_dir: str = "outputs"

    # Training
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use num_train_epochs
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1

    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"

    # Logging and evaluation
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: Optional[int] = 3

    # Mixed precision and efficiency
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0

    # Reproducibility
    seed: int = 42

    # Distributed training
    local_rank: int = -1
    ddp_find_unused_parameters: bool = False

    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0

    # Model selection
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

    # Checkpointing
    resume_from_checkpoint: Optional[str] = None

    # MLflow tracking
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "rosetta-transformer"
    mlflow_run_name: Optional[str] = None
    log_model_every_n_steps: int = 0  # 0 to disable

    def __post_init__(self):
        """Validate and process arguments."""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Validate precision settings
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")

        # Calculate warmup steps from ratio if specified
        if self.warmup_ratio > 0:
            if self.max_steps > 0:
                self.warmup_steps = int(self.max_steps * self.warmup_ratio)
            else:
                logger.warning(
                    "warmup_ratio is set but max_steps is not. "
                    "warmup_steps will be calculated after determining total steps."
                )

        # Set seed
        set_seed(self.seed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json_string(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save arguments to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write(self.to_json_string())

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TrainingArguments":
        """Load arguments from JSON file."""
        with open(filepath, "r") as f:
            args_dict = json.load(f)
        return cls(**args_dict)


class BaseModel(ABC, nn.Module):
    """Abstract base class for all models.

    Provides standard interface for model loading, saving, and forward pass.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base model.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config or {}

    @abstractmethod
    def forward(self, **kwargs) -> ModelOutput:
        """Forward pass through the model.

        Returns:
            ModelOutput containing loss, logits, and optional hidden states
        """
        pass

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save model to directory.

        Args:
            save_directory: Directory to save model files
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        model_path = save_directory / "pytorch_model.bin"
        torch.save(self.state_dict(), model_path)

        # Save config
        config_path = save_directory / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def load_pretrained(cls, load_directory: Union[str, Path]) -> "BaseModel":
        """Load model from directory.

        Args:
            load_directory: Directory containing model files

        Returns:
            Loaded model instance
        """
        load_directory = Path(load_directory)

        # Load config
        config_path = load_directory / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Initialize model
        model = cls(config=config)

        # Load state dict
        model_path = load_directory / "pytorch_model.bin"
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        logger.info(f"Model loaded from {load_directory}")
        return model

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Get number of model parameters.

        Args:
            only_trainable: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


class BaseTrainer:
    """Base trainer class with training loop, early stopping, and checkpointing.

    This class provides a standard training loop with the following features:
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Checkpointing with best model tracking
    - MLflow logging
    """

    def __init__(
        self,
        model: Union[BaseModel, PreTrainedModel],
        args: TrainingArguments,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[callable] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            args: Training arguments
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
            compute_metrics: Function to compute metrics from predictions
        """
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics

        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = None
        self.lr_scheduler = None

        # Setup mixed precision
        self.scaler = None
        if args.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.best_model_checkpoint = None
        self.early_stopping_counter = 0

        # MLflow
        self.mlflow_run = None
        if args.mlflow_tracking_uri:
            self._setup_mlflow()

    def _setup_device(self) -> torch.device:
        """Setup device for training."""
        if self.args.local_rank == -1:
            # Single GPU or CPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Distributed training
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
            torch.distributed.init_process_group(backend="nccl")

        return device

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.args.mlflow_tracking_uri)
        mlflow.set_experiment(self.args.mlflow_experiment_name)

        self.mlflow_run = mlflow.start_run(run_name=self.args.mlflow_run_name)

        # Log training arguments
        mlflow.log_params(self.args.to_dict())

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """Create optimizer and learning rate scheduler.

        Args:
            num_training_steps: Total number of training steps
        """
        # Create optimizer
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

        # Create scheduler
        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train(self) -> Dict[str, float]:
        """Run training loop.

        Returns:
            Dictionary of final metrics
        """
        if self.train_dataloader is None:
            raise ValueError("train_dataloader must be provided")

        # Calculate training steps
        num_update_steps_per_epoch = (
            len(self.train_dataloader) // self.args.gradient_accumulation_steps
        )

        if self.args.max_steps > 0:
            max_steps = self.args.max_steps
            num_train_epochs = max_steps // num_update_steps_per_epoch + 1
        else:
            max_steps = num_update_steps_per_epoch * self.args.num_train_epochs
            num_train_epochs = self.args.num_train_epochs

        # Create optimizer and scheduler
        self.create_optimizer_and_scheduler(max_steps)

        # Resume from checkpoint if specified
        if self.args.resume_from_checkpoint:
            self._load_checkpoint(self.args.resume_from_checkpoint)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num epochs = {num_train_epochs}")
        logger.info(
            f"  Batch size per device = {self.args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Gradient accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps}")

        # Training loop
        self.model.train()

        for epoch in range(num_train_epochs):
            self.epoch = epoch

            epoch_iterator = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_train_epochs}",
                disable=self.args.local_rank not in [-1, 0],
            )

            for step, batch in enumerate(epoch_iterator):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                if self.args.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)

                loss = outputs.loss / self.args.gradient_accumulation_steps

                # Backward pass
                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update weights
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.args.fp16:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm,
                    )

                    # Optimizer step
                    if self.args.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.args.logging_steps == 0:
                        metrics = {
                            "loss": loss.item() * self.args.gradient_accumulation_steps,
                            "learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": self.global_step,
                        }

                        if self.mlflow_run:
                            mlflow.log_metrics(metrics, step=self.global_step)

                        epoch_iterator.set_postfix(**metrics)

                    # Evaluation
                    if (
                        self.eval_dataloader
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate()

                        # Early stopping
                        if self._check_early_stopping(eval_metrics):
                            logger.info("Early stopping triggered")
                            if self.mlflow_run:
                                mlflow.end_run()
                            return eval_metrics

                        self.model.train()

                    # Checkpointing
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint()

                    # Log model to MLflow
                    if (
                        self.args.log_model_every_n_steps > 0
                        and self.global_step % self.args.log_model_every_n_steps == 0
                    ):
                        if self.mlflow_run:
                            mlflow.pytorch.log_model(
                                self.model, f"model_step_{self.global_step}"
                            )

                    # Check if max steps reached
                    if self.global_step >= max_steps:
                        break

            if self.global_step >= max_steps:
                break

        # Final evaluation
        final_metrics = {}
        if self.eval_dataloader:
            final_metrics = self.evaluate()

        # Load best model if requested
        if self.args.load_best_model_at_end and self.best_model_checkpoint:
            self._load_checkpoint(self.best_model_checkpoint)

        # End MLflow run
        if self.mlflow_run:
            if self.args.load_best_model_at_end:
                mlflow.pytorch.log_model(self.model, "best_model")
            mlflow.end_run()

        return final_metrics

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation loop.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}

        logger.info("***** Running evaluation *****")

        self.model.eval()

        all_losses = []
        all_predictions = []
        all_labels = []

        eval_iterator = tqdm(
            self.eval_dataloader,
            desc="Evaluation",
            disable=self.args.local_rank not in [-1, 0],
        )

        with torch.no_grad():
            for batch in eval_iterator:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                outputs = self.model(**batch)

                if outputs.loss is not None:
                    all_losses.append(outputs.loss.item())

                # Store predictions for metrics computation
                if self.compute_metrics:
                    if outputs.logits is not None:
                        all_predictions.append(outputs.logits.cpu())

                    if "labels" in batch:
                        all_labels.append(batch["labels"].cpu())

        # Compute metrics
        metrics = {}

        if all_losses:
            metrics["eval_loss"] = np.mean(all_losses)

        if self.compute_metrics and all_predictions and all_labels:
            predictions = torch.cat(all_predictions, dim=0)
            labels = torch.cat(all_labels, dim=0)

            computed_metrics = self.compute_metrics(predictions, labels)
            metrics.update({f"eval_{k}": v for k, v in computed_metrics.items()})

        # Log metrics
        logger.info(f"Evaluation metrics: {metrics}")

        if self.mlflow_run:
            mlflow.log_metrics(metrics, step=self.global_step)

        return metrics

    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria is met.

        Args:
            metrics: Current evaluation metrics

        Returns:
            True if training should stop
        """
        if self.args.early_stopping_patience is None:
            return False

        metric_key = f"eval_{self.args.metric_for_best_model}"

        if metric_key not in metrics:
            return False

        current_metric = metrics[metric_key]

        # First evaluation
        if self.best_metric is None:
            self.best_metric = current_metric
            return False

        # Check if improved
        if self.args.greater_is_better:
            improved = (
                current_metric > self.best_metric + self.args.early_stopping_threshold
            )
        else:
            improved = (
                current_metric < self.best_metric - self.args.early_stopping_threshold
            )

        if improved:
            self.best_metric = current_metric
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        return self.early_stopping_counter >= self.args.early_stopping_patience

    def _save_checkpoint(self, checkpoint_dir: Optional[str] = None) -> None:
        """Save training checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint (if None, auto-generates)
        """
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                self.args.output_dir,
                f"checkpoint-{self.global_step}",
            )

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        if isinstance(self.model, PreTrainedModel):
            self.model.save_pretrained(checkpoint_dir)
        else:
            self.model.save_pretrained(checkpoint_dir)

        # Save optimizer and scheduler
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler else None,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_metric": self.best_metric,
            },
            os.path.join(checkpoint_dir, "trainer_state.pt"),
        )

        # Save training arguments
        self.args.save(os.path.join(checkpoint_dir, "training_args.json"))

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

        # Update best model checkpoint
        if (
            self.best_model_checkpoint is None
            or checkpoint_dir == self.best_model_checkpoint
        ):
            self.best_model_checkpoint = checkpoint_dir

        # Delete old checkpoints if limit is set
        if self.args.save_total_limit is not None:
            self._delete_old_checkpoints()

    def _load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint
        """
        logger.info(f"Loading checkpoint from {checkpoint_dir}")

        # Load model
        if isinstance(self.model, PreTrainedModel):
            # For HuggingFace models
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(checkpoint_dir)
        else:
            self.model = self.model.load_pretrained(checkpoint_dir)

        self.model.to(self.device)

        # Load trainer state
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")

        if os.path.exists(trainer_state_path):
            state = torch.load(trainer_state_path, map_location=self.device)

            if self.optimizer:
                self.optimizer.load_state_dict(state["optimizer"])

            if self.lr_scheduler:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])

            if self.scaler and state["scaler"]:
                self.scaler.load_state_dict(state["scaler"])

            self.global_step = state.get("global_step", 0)
            self.epoch = state.get("epoch", 0)
            self.best_metric = state.get("best_metric")

    def _delete_old_checkpoints(self) -> None:
        """Delete old checkpoints to maintain save_total_limit."""
        checkpoints = []

        for item in os.listdir(self.args.output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_path = os.path.join(self.args.output_dir, item)
                if os.path.isdir(checkpoint_path):
                    checkpoints.append(checkpoint_path)

        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

        # Delete oldest checkpoints
        while len(checkpoints) > self.args.save_total_limit:
            checkpoint_to_delete = checkpoints.pop(0)

            # Don't delete best model
            if checkpoint_to_delete != self.best_model_checkpoint:
                logger.info(f"Deleting old checkpoint: {checkpoint_to_delete}")
                shutil.rmtree(checkpoint_to_delete)
