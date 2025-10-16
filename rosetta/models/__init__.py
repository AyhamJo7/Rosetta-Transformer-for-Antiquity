"""Model modules for Rosetta Transformer.

This package provides complete model training infrastructure including:
- Base classes for models and trainers
- Domain-adaptive pretraining for ancient texts
- Token classification for NER, POS tagging, and relation extraction
- Sequence-to-sequence models for translation and transliteration
"""

# Base classes
from rosetta.models.base import (
    BaseModel,
    BaseTrainer,
    ModelOutput,
    TrainingArguments,
)

# Pretraining
from rosetta.models.pretraining import (
    DataCollatorForAncientTextMLM,
    DomainPretrainer,
    PretrainingArguments,
    VocabularyExpander,
)

# Sequence-to-sequence tasks
from rosetta.models.seq2seq import (
    ConstrainedBeamSearch,
    Seq2SeqArguments,
    Seq2SeqModel,
    TransliterationTrainer,
    beam_search_decode,
    compute_bleu_with_bootstrap,
    compute_translation_metrics,
)

# Token classification tasks
from rosetta.models.token_tasks import (
    ConditionalRandomField,
    FocalLoss,
    RelationExtractionModel,
    TokenClassificationModel,
    TokenTaskArguments,
    TokenTaskTrainer,
)

__all__ = [
    # Base
    "BaseModel",
    "BaseTrainer",
    "ModelOutput",
    "TrainingArguments",
    # Pretraining
    "DomainPretrainer",
    "PretrainingArguments",
    "VocabularyExpander",
    "DataCollatorForAncientTextMLM",
    # Token tasks
    "TokenClassificationModel",
    "RelationExtractionModel",
    "TokenTaskArguments",
    "TokenTaskTrainer",
    "FocalLoss",
    "ConditionalRandomField",
    # Seq2seq
    "Seq2SeqModel",
    "Seq2SeqArguments",
    "TransliterationTrainer",
    "compute_translation_metrics",
    "compute_bleu_with_bootstrap",
    "beam_search_decode",
    "ConstrainedBeamSearch",
]
