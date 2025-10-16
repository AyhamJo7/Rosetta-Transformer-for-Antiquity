"""Sequence-to-sequence models for translation and transliteration.

This module provides models and trainers for translation and transliteration
tasks on ancient texts, with support for mBART, mT5, and custom metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from rosetta.models.base import BaseModel, BaseTrainer, ModelOutput, TrainingArguments
from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Seq2SeqArguments(TrainingArguments):
    """Extended training arguments for sequence-to-sequence tasks.

    Attributes:
        max_source_length: Maximum length of source sequences
        max_target_length: Maximum length of target sequences
        num_beams: Number of beams for beam search
        length_penalty: Length penalty for beam search
        no_repeat_ngram_size: Size of n-grams that can only occur once
        early_stopping: Whether to stop generation when all beams are finished
        generation_max_length: Maximum length for generation
        predict_with_generate: Whether to use generate() for evaluation
        label_smoothing: Label smoothing factor
        forced_bos_token_id: Token ID to force as first generated token
        source_lang: Source language code
        target_lang: Target language code
    """

    # Sequence lengths
    max_source_length: int = 512
    max_target_length: int = 512

    # Generation
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    generation_max_length: Optional[int] = None
    predict_with_generate: bool = True

    # Training
    label_smoothing: float = 0.1
    forced_bos_token_id: Optional[int] = None

    # Languages
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        super().__post_init__()

        if self.generation_max_length is None:
            self.generation_max_length = self.max_target_length


class Seq2SeqModel(BaseModel):
    """Wrapper for sequence-to-sequence models.

    Supports mBART, mT5, and other encoder-decoder transformers
    for translation and transliteration tasks.
    """

    def __init__(
        self,
        model_name: str = "facebook/mbart-large-50",
        source_lang: str = "en_XX",
        target_lang: str = "ar_AR",
        config: Optional[Dict] = None,
    ):
        """Initialize seq2seq model.

        Args:
            model_name: Name or path of pretrained model
            source_lang: Source language code
            target_lang: Target language code
            config: Additional config dict
        """
        super().__init__(config)

        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Load model
        logger.info(f"Loading seq2seq model from {model_name}...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set language tokens for mBART
        if "mbart" in model_name.lower():
            self.tokenizer.src_lang = source_lang
            self.tokenizer.tgt_lang = target_lang

        # Store config
        self.config.update(
            {
                "model_name": model_name,
                "source_lang": source_lang,
                "target_lang": target_lang,
            }
        )

        logger.info(
            f"Initialized seq2seq model with {self.num_parameters():,} parameters"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
        """Forward pass.

        Args:
            input_ids: Source token IDs
            attention_mask: Source attention mask
            labels: Target token IDs (for training)
            decoder_input_ids: Decoder input IDs
            decoder_attention_mask: Decoder attention mask
            **kwargs: Additional arguments

        Returns:
            ModelOutput with loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
            **kwargs,
        )

        return ModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate translations/transliterations.

        Args:
            input_ids: Source token IDs
            attention_mask: Source attention mask
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            length_penalty: Length penalty
            no_repeat_ngram_size: N-gram blocking size
            early_stopping: Early stopping flag
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        # Set decoder start token for mBART
        forced_bos_token_id = None
        if "mbart" in self.model_name.lower():
            forced_bos_token_id = self.tokenizer.lang_code_to_id[self.target_lang]

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            forced_bos_token_id=forced_bos_token_id,
            **kwargs,
        )

        return generated_ids

    def translate(
        self,
        texts: List[str],
        max_length: int = 512,
        num_beams: int = 4,
        batch_size: int = 8,
        **kwargs,
    ) -> List[str]:
        """Translate texts.

        Args:
            texts: List of source texts
            max_length: Maximum generation length
            num_beams: Number of beams
            batch_size: Batch size for processing
            **kwargs: Additional generation arguments

        Returns:
            List of translated texts
        """
        self.eval()
        device = next(self.parameters()).device

        translations = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Generate
                generated_ids = self.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    num_beams=num_beams,
                    **kwargs,
                )

                # Decode
                batch_translations = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                translations.extend(batch_translations)

        return translations


class TransliterationTrainer(BaseTrainer):
    """Specialized trainer for transliteration tasks.

    Extends BaseTrainer with transliteration-specific features:
    - Character-level metrics
    - Beam search evaluation
    - Custom decoding strategies
    """

    def __init__(
        self,
        model: Union[Seq2SeqModel, PreTrainedModel],
        args: Seq2SeqArguments,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[callable] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """Initialize transliteration trainer.

        Args:
            model: Model to train
            args: Training arguments
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            compute_metrics: Metrics computation function
            tokenizer: Tokenizer for decoding
        """
        super().__init__(
            model, args, train_dataloader, eval_dataloader, compute_metrics
        )

        self.tokenizer = tokenizer

        if tokenizer is None and hasattr(model, "tokenizer"):
            self.tokenizer = model.tokenizer

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model with generation.

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.args.predict_with_generate:
            return super().evaluate()

        if self.eval_dataloader is None:
            return {}

        logger.info("***** Running evaluation with generation *****")

        self.model.eval()

        all_predictions = []
        all_labels = []
        all_losses = []

        with torch.no_grad():
            for batch in self.eval_dataloader:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Compute loss
                outputs = self.model(**batch)
                if outputs.loss is not None:
                    all_losses.append(outputs.loss.item())

                # Generate predictions
                if isinstance(self.model, Seq2SeqModel):
                    generated_ids = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        max_length=self.args.generation_max_length,
                        num_beams=self.args.num_beams,
                        length_penalty=self.args.length_penalty,
                        no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                        early_stopping=self.args.early_stopping,
                    )
                else:
                    generated_ids = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        max_length=self.args.generation_max_length,
                        num_beams=self.args.num_beams,
                    )

                all_predictions.extend(generated_ids.cpu().numpy())

                if "labels" in batch:
                    labels = batch["labels"].cpu().numpy()
                    # Replace -100 with pad token
                    labels = np.where(
                        labels != -100, labels, self.tokenizer.pad_token_id
                    )
                    all_labels.extend(labels)

        # Compute metrics
        metrics = {}

        if all_losses:
            metrics["eval_loss"] = np.mean(all_losses)

        if self.compute_metrics and all_predictions and all_labels:
            # Decode predictions and labels
            decoded_preds = self.tokenizer.batch_decode(
                all_predictions, skip_special_tokens=True
            )

            decoded_labels = self.tokenizer.batch_decode(
                all_labels, skip_special_tokens=True
            )

            # Compute metrics
            computed_metrics = self.compute_metrics(decoded_preds, decoded_labels)
            metrics.update({f"eval_{k}": v for k, v in computed_metrics.items()})

        # Log metrics
        logger.info(f"Evaluation metrics: {metrics}")

        if self.mlflow_run:
            import mlflow

            mlflow.log_metrics(metrics, step=self.global_step)

        return metrics


def compute_translation_metrics(
    predictions: List[str],
    references: List[str],
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> Dict[str, float]:
    """Compute translation/transliteration metrics.

    Computes BLEU, chrF, and exact match scores.

    Args:
        predictions: Predicted texts
        references: Reference texts
        tokenizer: Optional tokenizer for additional metrics

    Returns:
        Dictionary of metrics
    """
    from sacrebleu import corpus_bleu, corpus_chrf

    metrics = {}

    # BLEU score
    try:
        # Format references for sacrebleu (list of lists)
        formatted_refs = [[ref] for ref in references]
        bleu = corpus_bleu(predictions, formatted_refs)
        metrics["bleu"] = bleu.score
    except Exception as e:
        logger.warning(f"Failed to compute BLEU: {e}")
        metrics["bleu"] = 0.0

    # chrF score
    try:
        chrf = corpus_chrf(predictions, references)
        metrics["chrf"] = chrf.score
    except Exception as e:
        logger.warning(f"Failed to compute chrF: {e}")
        metrics["chrf"] = 0.0

    # Exact match
    exact_matches = sum(
        1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip()
    )
    metrics["exact_match"] = 100.0 * exact_matches / len(predictions)

    # Character-level accuracy
    total_chars = 0
    correct_chars = 0

    for pred, ref in zip(predictions, references):
        total_chars += len(ref)
        # Count matching characters at same positions
        for p_char, r_char in zip(pred, ref):
            if p_char == r_char:
                correct_chars += 1

    if total_chars > 0:
        metrics["char_accuracy"] = 100.0 * correct_chars / total_chars
    else:
        metrics["char_accuracy"] = 0.0

    return metrics


def compute_bleu_with_bootstrap(
    predictions: List[str],
    references: List[str],
    num_samples: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute BLEU score with bootstrapped confidence intervals.

    Args:
        predictions: Predicted texts
        references: Reference texts
        num_samples: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (BLEU score, lower bound, upper bound)
    """
    from sacrebleu import corpus_bleu

    n = len(predictions)

    # Compute main BLEU score
    formatted_refs = [[ref] for ref in references]
    main_bleu = corpus_bleu(predictions, formatted_refs).score

    # Bootstrap sampling
    bleu_scores = []

    for _ in range(num_samples):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)

        sampled_preds = [predictions[i] for i in indices]
        sampled_refs = [[references[i]] for i in indices]

        try:
            bleu = corpus_bleu(sampled_preds, sampled_refs).score
            bleu_scores.append(bleu)
        except:
            continue

    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bleu_scores, lower_percentile)
    upper_bound = np.percentile(bleu_scores, upper_percentile)

    return main_bleu, lower_bound, upper_bound


def beam_search_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_beams: int = 4,
    max_length: int = 512,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
) -> List[List[int]]:
    """Custom beam search decoding implementation.

    Args:
        model: Seq2seq model
        input_ids: Source token IDs (batch, seq_len)
        attention_mask: Source attention mask
        num_beams: Number of beams
        max_length: Maximum generation length
        length_penalty: Length penalty factor
        early_stopping: Whether to stop when all beams finish

    Returns:
        List of decoded sequences (one per input)
    """
    device = input_ids.device
    batch_size = input_ids.size(0)

    # Expand inputs for beam search
    expanded_input_ids = (
        input_ids.unsqueeze(1).repeat(1, num_beams, 1).view(-1, input_ids.size(-1))
    )

    if attention_mask is not None:
        expanded_attention_mask = (
            attention_mask.unsqueeze(1)
            .repeat(1, num_beams, 1)
            .view(-1, attention_mask.size(-1))
        )
    else:
        expanded_attention_mask = None

    # Use model's generate method (which implements beam search)
    if hasattr(model, "generate"):
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
        )

        return generated.tolist()

    # Fallback: return input (this should not happen with proper models)
    return input_ids.tolist()


class ConstrainedBeamSearch:
    """Beam search with lexical constraints.

    Allows enforcing that certain tokens or phrases appear in the output,
    useful for transliteration where certain character mappings are known.
    """

    def __init__(
        self,
        required_tokens: Optional[List[List[int]]] = None,
        forbidden_tokens: Optional[List[int]] = None,
    ):
        """Initialize constrained beam search.

        Args:
            required_tokens: List of token sequences that must appear
            forbidden_tokens: List of forbidden token IDs
        """
        self.required_tokens = required_tokens or []
        self.forbidden_tokens = set(forbidden_tokens or [])

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Process scores to enforce constraints.

        Args:
            input_ids: Current sequence (batch * num_beams, seq_len)
            scores: Next token scores (batch * num_beams, vocab_size)

        Returns:
            Modified scores
        """
        # Forbid certain tokens
        if self.forbidden_tokens:
            for token_id in self.forbidden_tokens:
                scores[:, token_id] = -float("inf")

        # TODO: Implement required tokens constraint
        # This is complex and requires tracking which required tokens
        # have been generated in each beam

        return scores
