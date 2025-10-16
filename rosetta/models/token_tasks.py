"""Token-level task models for NER, POS tagging, and relation extraction.

This module provides model classes for token classification tasks on ancient texts,
including Named Entity Recognition (NER), Part-of-Speech (POS) tagging, and
Relation Extraction (RE) with class imbalance handling.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    PreTrainedModel,
)

from rosetta.models.base import BaseModel, BaseTrainer, ModelOutput, TrainingArguments
from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TokenTaskArguments(TrainingArguments):
    """Extended training arguments for token classification tasks.

    Attributes:
        label_smoothing: Label smoothing factor (0.0 = no smoothing)
        use_crf: Whether to use CRF layer on top
        use_focal_loss: Whether to use focal loss for class imbalance
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
        class_weights: Optional class weights for imbalanced datasets
        return_entity_level_metrics: Whether to compute entity-level metrics
    """

    label_smoothing: float = 0.0
    use_crf: bool = False
    use_focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    class_weights: Optional[List[float]] = None
    return_entity_level_metrics: bool = True


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the class.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Initialize focal loss.

        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
            reduction: Specifies the reduction to apply to the output
            ignore_index: Specifies a target value that is ignored
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predictions (logits) of shape (batch, num_classes)
            targets: Ground truth labels of shape (batch,)

        Returns:
            Focal loss value
        """
        # Get probabilities
        p = F.softmax(inputs, dim=-1)

        # Create one-hot encoding of targets
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", ignore_index=self.ignore_index
        )

        # Get probability of true class
        p_t = p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # Apply ignore_index mask
        mask = targets != self.ignore_index
        p_t = p_t * mask

        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.sum() / mask.sum()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ConditionalRandomField(nn.Module):
    """Conditional Random Field (CRF) layer for sequence tagging.

    Implements linear-chain CRF for enforcing tag sequence constraints.
    """

    def __init__(self, num_tags: int, batch_first: bool = True):
        """Initialize CRF layer.

        Args:
            num_tags: Number of tags
            batch_first: Whether batch dimension is first
        """
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        # Transition parameters: transitions[i, j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            emissions: Emission scores of shape (batch, seq_len, num_tags)
            tags: Target tags of shape (batch, seq_len)
            mask: Mask of shape (batch, seq_len)

        Returns:
            Negative log-likelihood
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)  # (seq_len, batch, num_tags)
            tags = tags.transpose(0, 1)  # (seq_len, batch)
            if mask is not None:
                mask = mask.transpose(0, 1)  # (seq_len, batch)

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        # Compute log partition function (forward algorithm)
        log_partition = self._compute_log_partition(emissions, mask)

        # Compute score of gold sequence
        gold_score = self._compute_score(emissions, tags, mask)

        # Return negative log-likelihood
        return torch.mean(log_partition - gold_score)

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Find most likely tag sequence using Viterbi algorithm.

        Args:
            emissions: Emission scores of shape (batch, seq_len, num_tags)
            mask: Mask of shape (batch, seq_len)

        Returns:
            List of decoded tag sequences
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        if mask is None:
            mask = torch.ones(
                emissions.shape[0],
                emissions.shape[1],
                dtype=torch.bool,
                device=emissions.device,
            )

        return self._viterbi_decode(emissions, mask)

    def _compute_log_partition(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute log partition function using forward algorithm."""
        seq_len, batch_size = mask.shape

        # Start with start transitions
        score = self.start_transitions + emissions[0]  # (batch, num_tags)

        for i in range(1, seq_len):
            # Broadcast for transitions
            broadcast_score = score.unsqueeze(2)  # (batch, num_tags, 1)
            broadcast_emissions = emissions[i].unsqueeze(1)  # (batch, 1, num_tags)
            broadcast_transitions = self.transitions.unsqueeze(
                0
            )  # (1, num_tags, num_tags)

            # Compute next scores
            next_score = broadcast_score + broadcast_emissions + broadcast_transitions

            # Log-sum-exp
            next_score = torch.logsumexp(next_score, dim=1)  # (batch, num_tags)

            # Apply mask
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # Add end transitions
        score = score + self.end_transitions

        # Log-sum-exp over all possible end tags
        return torch.logsumexp(score, dim=1)

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute score of a given tag sequence."""
        seq_len, batch_size = mask.shape

        # Start transitions
        score = self.start_transitions[tags[0]]  # (batch,)

        # Add emission and transition scores
        for i in range(seq_len):
            # Emission score
            score = (
                score
                + emissions[i].gather(1, tags[i].unsqueeze(1)).squeeze(1) * mask[i]
            )

            # Transition score (except for last timestep)
            if i < seq_len - 1:
                score = score + self.transitions[tags[i], tags[i + 1]] * mask[i + 1]

        # Get last valid tags
        last_tags = tags[mask.sum(0).long() - 1, torch.arange(batch_size)]

        # Add end transitions
        score = score + self.end_transitions[last_tags]

        return score

    def _viterbi_decode(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> List[List[int]]:
        """Viterbi algorithm for finding best tag sequence."""
        seq_len, batch_size, num_tags = emissions.shape

        # Initialize
        score = self.start_transitions + emissions[0]
        history = []

        # Forward pass
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            broadcast_transitions = self.transitions.unsqueeze(0)

            next_score = broadcast_score + broadcast_emissions + broadcast_transitions

            # Get best previous tags
            next_score, indices = next_score.max(dim=1)

            # Apply mask
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Add end transitions
        score = score + self.end_transitions

        # Backtrack
        best_tags_list = []

        for batch_idx in range(batch_size):
            # Find best last tag
            seq_ends = mask[:, batch_idx].sum().long()
            _, best_last_tag = score[batch_idx].max(dim=0)

            best_tags = [best_last_tag.item()]

            # Backtrack through history
            for hist_idx in reversed(range(seq_ends - 1)):
                best_last_tag = history[hist_idx][batch_idx, best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class TokenClassificationModel(BaseModel):
    """Model for token classification (NER, POS tagging, etc.).

    Implements a transformer encoder with a classification head for
    token-level prediction tasks.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = 9,  # Default for common NER (O, B-PER, I-PER, etc.)
        use_crf: bool = False,
        dropout: float = 0.1,
        config: Optional[Dict] = None,
    ):
        """Initialize token classification model.

        Args:
            model_name: Name or path of pretrained model
            num_labels: Number of labels/tags
            use_crf: Whether to use CRF layer
            dropout: Dropout rate
            config: Additional config dict
        """
        super().__init__(config)

        self.num_labels = num_labels
        self.use_crf = use_crf

        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Optional CRF layer
        self.crf: Optional[ConditionalRandomField]
        if use_crf:
            self.crf = ConditionalRandomField(num_labels)
        else:
            self.crf = None

        # Store config
        self.config.update(
            {
                "model_name": model_name,
                "num_labels": num_labels,
                "use_crf": use_crf,
                "dropout": dropout,
                "hidden_size": hidden_size,
            }
        )

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
        """Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels (optional)
            **kwargs: Additional arguments

        Returns:
            ModelOutput with loss and logits
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        sequence_output = self.dropout(sequence_output)

        # Classify
        logits = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

        loss = None
        if labels is not None:
            if self.use_crf and self.crf is not None:
                # Use CRF loss
                if attention_mask is not None:
                    loss = self.crf(logits, labels, attention_mask.bool())
                else:
                    # Create a default mask if None
                    batch_size, seq_len = labels.shape
                    mask = torch.ones(
                        batch_size, seq_len, dtype=torch.bool, device=labels.device
                    )
                    loss = self.crf(logits, labels, mask)
            else:
                # Use cross-entropy loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ModelOutput(loss=loss, logits=logits)

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Predict tags for input.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            List of predicted tag sequences
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)

            if self.use_crf and self.crf is not None:
                if attention_mask is not None:
                    predictions = self.crf.decode(outputs.logits, attention_mask.bool())
                else:
                    predictions = self.crf.decode(outputs.logits, None)
            else:
                if outputs.logits is not None:
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    predictions = predictions.tolist()
                else:
                    predictions = []

        return predictions


class RelationExtractionModel(BaseModel):
    """Model for relation extraction between entities.

    Implements entity-pair classification for identifying semantic
    relations between marked entities in text.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_relations: int = 10,
        dropout: float = 0.1,
        pooling: str = "entity_markers",
        config: Optional[Dict] = None,
    ):
        """Initialize relation extraction model.

        Args:
            model_name: Name or path of pretrained model
            num_relations: Number of relation types
            dropout: Dropout rate
            pooling: Pooling strategy ('entity_markers', 'entity_start', 'cls')
            config: Additional config dict
        """
        super().__init__(config)

        self.num_relations = num_relations
        self.pooling = pooling

        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Relation classification head
        # Concatenate head and tail entity representations
        classifier_input_size = hidden_size * 2

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_relations),
        )

        # Store config
        self.config.update(
            {
                "model_name": model_name,
                "num_relations": num_relations,
                "dropout": dropout,
                "pooling": pooling,
                "hidden_size": hidden_size,
            }
        )

    def pool_entity_representation(
        self,
        sequence_output: torch.Tensor,
        entity_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Pool entity representation from sequence.

        Args:
            sequence_output: Sequence output (batch, seq_len, hidden)
            entity_positions: Entity positions (batch, 2) with [start, end] indices

        Returns:
            Pooled entity representations (batch, hidden)
        """
        batch_size = sequence_output.size(0)

        # Gather entity span representations
        entity_reps = []

        for i in range(batch_size):
            start, end = entity_positions[i]

            # Get span representation
            if self.pooling == "entity_start":
                entity_rep = sequence_output[i, start]
            elif self.pooling == "entity_markers":
                # Average over entity span
                entity_rep = sequence_output[i, start : end + 1].mean(dim=0)
            else:  # cls
                entity_rep = sequence_output[i, 0]

            entity_reps.append(entity_rep)

        return torch.stack(entity_reps)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_positions: Optional[torch.Tensor] = None,
        tail_positions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
        """Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            head_positions: Head entity positions (batch, 2)
            tail_positions: Tail entity positions (batch, 2)
            labels: Relation labels (optional)
            **kwargs: Additional arguments

        Returns:
            ModelOutput with loss and logits
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        # Pool entity representations
        head_rep = self.pool_entity_representation(sequence_output, head_positions)
        tail_rep = self.pool_entity_representation(sequence_output, tail_positions)

        # Concatenate head and tail
        pair_rep = torch.cat([head_rep, tail_rep], dim=-1)

        # Classify relation
        pair_rep = self.dropout(pair_rep)
        logits = self.classifier(pair_rep)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return ModelOutput(loss=loss, logits=logits)

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_positions: Optional[torch.Tensor] = None,
        tail_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict relations.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            head_positions: Head entity positions
            tail_positions: Tail entity positions

        Returns:
            Predicted relation labels
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_positions=head_positions,
                tail_positions=tail_positions,
            )

            predictions = torch.argmax(outputs.logits, dim=-1)

        return predictions


class TokenTaskTrainer(BaseTrainer):
    """Trainer for token classification tasks with class imbalance handling."""

    def __init__(
        self,
        model: Union[BaseModel, PreTrainedModel],
        args: TokenTaskArguments,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            args: Training arguments
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            compute_metrics: Metrics computation function
        """
        super().__init__(
            model, args, train_dataloader, eval_dataloader, compute_metrics
        )

        # Setup focal loss if requested
        # Note: TokenTaskArguments may have these attributes, but mypy doesn't see them
        # through TrainingArguments, so we need type guards
        token_args = args  # Type hint: TokenTaskArguments
        if hasattr(token_args, "use_focal_loss") and token_args.use_focal_loss:  # type: ignore[attr-defined]
            logger.info(
                f"Using focal loss (alpha={token_args.focal_alpha}, gamma={token_args.focal_gamma})"  # type: ignore[attr-defined]
            )

    def compute_loss(self, model: nn.Module, inputs: Dict) -> torch.Tensor:
        """Compute loss with optional focal loss or class weights.

        Args:
            model: Model
            inputs: Input batch

        Returns:
            Loss tensor
        """
        outputs = model(**inputs)

        if outputs.loss is not None:
            return outputs.loss

        # Manual loss computation for focal loss
        token_args = self.args  # Type: TokenTaskArguments
        if hasattr(token_args, "use_focal_loss") and token_args.use_focal_loss:  # type: ignore[attr-defined]
            focal_loss_fn = FocalLoss(
                alpha=token_args.focal_alpha,  # type: ignore[attr-defined]
                gamma=token_args.focal_gamma,  # type: ignore[attr-defined]
            )

            loss = focal_loss_fn(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                inputs["labels"].view(-1),
            )
        else:
            class_weights = getattr(token_args, "class_weights", None)
            loss_fct = nn.CrossEntropyLoss(
                weight=(
                    torch.tensor(class_weights).to(self.device)
                    if class_weights
                    else None
                )
            )

            loss = loss_fct(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                inputs["labels"].view(-1),
            )

        return loss
