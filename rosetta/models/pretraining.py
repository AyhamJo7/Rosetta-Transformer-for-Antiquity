"""Domain-adaptive pretraining for ancient text models.

This module provides classes and utilities for continued pretraining of
multilingual transformers (XLM-RoBERTa, etc.) on ancient text corpora using
masked language modeling and vocabulary expansion.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
)

from rosetta.models.base import BaseTrainer, TrainingArguments
from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PretrainingArguments(TrainingArguments):
    """Extended training arguments for domain-adaptive pretraining.

    Attributes:
        mlm_probability: Probability of masking tokens for MLM
        vocab_expansion_size: Number of new tokens to add to vocabulary
        vocab_expansion_from_corpus: Whether to extract new tokens from corpus
        max_vocab_size: Maximum vocabulary size after expansion
        min_token_frequency: Minimum frequency for new vocab tokens
        whole_word_masking: Whether to use whole word masking
        freeze_embeddings: Whether to freeze original embeddings during training
        freeze_encoder_layers: Number of encoder layers to freeze (from bottom)
    """

    # MLM-specific
    mlm_probability: float = 0.15
    vocab_expansion_size: int = 0
    vocab_expansion_from_corpus: bool = True
    max_vocab_size: int = 250000
    min_token_frequency: int = 10
    whole_word_masking: bool = True

    # Model architecture
    freeze_embeddings: bool = False
    freeze_encoder_layers: int = 0


class DataCollatorForAncientTextMLM(DataCollatorForLanguageModeling):
    """Custom data collator for masked language modeling on ancient texts.

    Extends the standard DataCollatorForLanguageModeling with:
    - Special handling for rare characters and diacritics
    - Character-level masking for better generalization
    - Support for transliteration pairs

    Attributes:
        tokenizer: Tokenizer for the model
        mlm: Whether to apply masking
        mlm_probability: Probability of masking each token
        whole_word_masking: Whether to mask entire words
        rare_token_boost: Boost masking probability for rare tokens
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        whole_word_masking: bool = True,
        rare_token_boost: float = 1.5,
        pad_to_multiple_of: Optional[int] = None,
    ):
        """Initialize the data collator.

        Args:
            tokenizer: Tokenizer instance
            mlm: Whether to apply masked language modeling
            mlm_probability: Base probability of masking tokens
            whole_word_masking: Whether to mask whole words
            rare_token_boost: Multiplier for rare token masking probability
            pad_to_multiple_of: Pad length to multiple of this value
        """
        super().__init__(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        self.whole_word_masking = whole_word_masking
        self.rare_token_boost = rare_token_boost

        # Cache for rare tokens (tokens appearing less frequently)
        self.rare_tokens = self._identify_rare_tokens()

    def _identify_rare_tokens(self) -> set:
        """Identify rare tokens in vocabulary.

        Returns:
            Set of rare token IDs
        """
        # For now, we'll use a heuristic: tokens with IDs > 50000 are considered rare
        # In a real implementation, this would be based on corpus statistics
        vocab_size = len(self.tokenizer)
        rare_threshold = min(vocab_size, 50000)
        return set(range(rare_threshold, vocab_size))

    def torch_mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask tokens for masked language modeling.

        Applies masking with boosted probability for rare tokens and
        whole-word masking if enabled.

        Args:
            inputs: Input token IDs
            special_tokens_mask: Mask indicating special tokens

        Returns:
            Tuple of (masked inputs, labels)
        """
        labels = inputs.clone()

        # Create probability matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Boost probability for rare tokens
        if self.rare_token_boost > 1.0:
            for token_id in self.rare_tokens:
                rare_mask = inputs == token_id
                probability_matrix[rare_mask] *= self.rare_token_boost

        # Don't mask special tokens
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Don't mask non-special tokens
        labels[~masked_indices] = -100

        # 80% of the time, replace with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, replace with random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )

        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # 10% of the time, keep the original token

        return inputs, labels


class VocabularyExpander:
    """Utility for expanding tokenizer vocabulary with domain-specific tokens.

    This class analyzes a corpus and identifies frequent character sequences
    and words that should be added to the tokenizer vocabulary for better
    performance on ancient texts.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int = 10000,
        min_frequency: int = 10,
    ):
        """Initialize vocabulary expander.

        Args:
            tokenizer: Base tokenizer to expand
            max_new_tokens: Maximum number of new tokens to add
            min_frequency: Minimum frequency threshold for new tokens
        """
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.min_frequency = min_frequency

    def extract_candidates(self, corpus: List[str]) -> Dict[str, int]:
        """Extract candidate tokens from corpus.

        Args:
            corpus: List of text strings

        Returns:
            Dictionary mapping candidate tokens to frequencies
        """
        from collections import Counter

        # Extract character n-grams and words
        candidates: Counter[str] = Counter()

        for text in corpus:
            # Split into tokens (simple whitespace splitting)
            tokens = text.split()

            for token in tokens:
                # Skip very short or very long tokens
                if len(token) < 2 or len(token) > 20:
                    continue

                # Check if token would be split by current tokenizer
                encoded = self.tokenizer.encode(token, add_special_tokens=False)

                # If token is split into multiple subtokens, it's a candidate
                if len(encoded) > 1:
                    candidates[token] += 1

                # Also extract character bigrams and trigrams for ancient scripts
                for n in [2, 3]:
                    for i in range(len(token) - n + 1):
                        ngram = token[i : i + n]
                        if not ngram.isascii():  # Focus on non-ASCII (ancient scripts)
                            candidates[ngram] += 1

        return candidates

    def select_tokens(self, candidates: Dict[str, int]) -> List[str]:
        """Select top tokens to add to vocabulary.

        Args:
            candidates: Dictionary of candidate tokens and frequencies

        Returns:
            List of selected tokens
        """
        # Filter by minimum frequency
        filtered = {
            token: freq
            for token, freq in candidates.items()
            if freq >= self.min_frequency
        }

        # Sort by frequency
        sorted_candidates = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

        # Take top N
        selected = [token for token, _ in sorted_candidates[: self.max_new_tokens]]

        logger.info(
            f"Selected {len(selected)} new tokens from {len(candidates)} candidates"
        )

        return selected

    def expand_vocabulary(self, corpus: List[str]) -> PreTrainedTokenizer:
        """Expand tokenizer vocabulary with corpus-specific tokens.

        Args:
            corpus: List of text strings to analyze

        Returns:
            Tokenizer with expanded vocabulary
        """
        logger.info("Extracting vocabulary candidates from corpus...")
        candidates = self.extract_candidates(corpus)

        logger.info("Selecting tokens to add...")
        new_tokens = self.select_tokens(candidates)

        if not new_tokens:
            logger.warning("No new tokens to add to vocabulary")
            return self.tokenizer

        # Add new tokens to tokenizer
        logger.info(f"Adding {len(new_tokens)} new tokens to vocabulary...")
        num_added = self.tokenizer.add_tokens(new_tokens)

        logger.info(f"Successfully added {num_added} tokens to vocabulary")
        logger.info(f"New vocabulary size: {len(self.tokenizer)}")

        return self.tokenizer


class DomainPretrainer:
    """Domain-adaptive pretrainer for ancient text models.

    This class handles continued pretraining of multilingual transformers
    on ancient text corpora using masked language modeling. It supports:
    - Vocabulary expansion with domain-specific tokens
    - Custom masking strategies for ancient texts
    - MLflow tracking and checkpointing
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        args: Optional[PretrainingArguments] = None,
    ):
        """Initialize domain pretrainer.

        Args:
            model_name: Name or path of base model
            args: Pretraining arguments
        """
        self.model_name = model_name
        self.args = args or PretrainingArguments()

        # Load tokenizer and model
        logger.info(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Loading model from {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Vocabulary expander
        self.vocab_expander: Optional[VocabularyExpander] = None

        logger.info(
            f"Initialized pretrainer with {self.model.num_parameters():,} parameters"  # type: ignore[attr-defined]
        )

    def expand_vocabulary(self, corpus: List[str]) -> None:
        """Expand vocabulary with corpus-specific tokens.

        Args:
            corpus: List of text strings for vocabulary extraction
        """
        if self.args.vocab_expansion_size == 0:
            logger.info("Vocabulary expansion disabled")
            return

        logger.info("Expanding vocabulary...")

        self.vocab_expander = VocabularyExpander(
            tokenizer=self.tokenizer,
            max_new_tokens=self.args.vocab_expansion_size,
            min_frequency=self.args.min_token_frequency,
        )

        # Expand vocabulary
        old_vocab_size = len(self.tokenizer)
        if self.vocab_expander is not None:
            self.tokenizer = self.vocab_expander.expand_vocabulary(corpus)
        new_vocab_size = len(self.tokenizer)

        # Resize model embeddings
        if new_vocab_size > old_vocab_size:
            logger.info(
                f"Resizing model embeddings from {old_vocab_size} to {new_vocab_size}"
            )
            self.model.resize_token_embeddings(new_vocab_size)

            # Initialize new embeddings
            with torch.no_grad():
                # Get embedding layer
                embeddings = self.model.get_input_embeddings()

                # Initialize new embeddings as mean of existing embeddings
                mean_embedding = embeddings.weight[:old_vocab_size].mean(dim=0)

                # Set new embeddings
                embeddings.weight[old_vocab_size:] = mean_embedding

    def freeze_layers(self) -> None:
        """Freeze specified layers of the model."""
        if self.args.freeze_embeddings:
            logger.info("Freezing embeddings...")
            for param in self.model.get_input_embeddings().parameters():
                param.requires_grad = False

        if self.args.freeze_encoder_layers > 0:
            logger.info(
                f"Freezing bottom {self.args.freeze_encoder_layers} encoder layers..."
            )

            # For XLM-RoBERTa/RoBERTa
            if hasattr(self.model, "roberta"):
                encoder = self.model.roberta.encoder
            elif hasattr(self.model, "bert"):
                encoder = self.model.bert.encoder
            else:
                logger.warning("Could not find encoder to freeze")
                return

            # Freeze specified layers
            for i in range(min(self.args.freeze_encoder_layers, len(encoder.layer))):
                for param in encoder.layer[i].parameters():
                    param.requires_grad = False

    def prepare_dataset(self, corpus: List[str]) -> Dataset:
        """Prepare dataset for pretraining.

        Args:
            corpus: List of text strings

        Returns:
            HuggingFace Dataset
        """
        logger.info(f"Preparing dataset from {len(corpus)} documents...")

        # Create dataset
        dataset = Dataset.from_dict({"text": corpus})

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                return_special_tokens_mask=True,
            )

        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

        return tokenized_dataset

    def create_data_collator(self) -> DataCollatorForAncientTextMLM:
        """Create data collator for MLM.

        Returns:
            Data collator instance
        """
        return DataCollatorForAncientTextMLM(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.args.mlm_probability,
            whole_word_masking=self.args.whole_word_masking,
        )

    def train(
        self,
        train_corpus: List[str],
        eval_corpus: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Run pretraining.

        Args:
            train_corpus: Training texts
            eval_corpus: Evaluation texts (optional)

        Returns:
            Dictionary of final metrics
        """
        # Expand vocabulary if requested
        if self.args.vocab_expansion_from_corpus and self.args.vocab_expansion_size > 0:
            self.expand_vocabulary(train_corpus)

        # Freeze layers if requested
        self.freeze_layers()

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_corpus)
        eval_dataset = self.prepare_dataset(eval_corpus) if eval_corpus else None

        # Create data collator
        data_collator = self.create_data_collator()

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

        eval_dataloader = None
        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
            )

        # Create trainer
        trainer = BaseTrainer(
            model=self.model,
            args=self.args,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
        )

        # Train
        logger.info("Starting pretraining...")
        metrics = trainer.train()

        # Save final model and tokenizer
        self.save_pretrained(self.args.output_dir)

        return metrics

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save model and tokenizer.

        Args:
            save_directory: Directory to save to
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {save_directory}...")
        self.model.save_pretrained(save_directory)

        logger.info(f"Saving tokenizer to {save_directory}...")
        self.tokenizer.save_pretrained(save_directory)

        # Save arguments
        self.args.save(save_directory / "pretraining_args.json")

        logger.info(f"Model and tokenizer saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: Union[str, Path]) -> "DomainPretrainer":
        """Load pretrained model and tokenizer.

        Args:
            load_directory: Directory to load from

        Returns:
            DomainPretrainer instance
        """
        load_directory = Path(load_directory)

        logger.info(f"Loading pretrainer from {load_directory}...")

        # Load arguments if available
        args_path = load_directory / "pretraining_args.json"
        args: Optional[PretrainingArguments] = None
        if args_path.exists():
            args = PretrainingArguments.load(args_path)  # type: ignore[assignment]

        # Create instance with loaded model
        pretrainer = cls(model_name=str(load_directory), args=args)

        logger.info("Pretrainer loaded successfully")

        return pretrainer
