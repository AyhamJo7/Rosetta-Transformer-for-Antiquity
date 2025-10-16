"""Evaluation framework for Rosetta Transformer.

This module provides comprehensive metrics and validators for evaluating
NER, relation extraction, and seq2seq tasks on ancient texts.
"""

from rosetta.evaluation.metrics import (
    bootstrap_confidence_interval,
    compute_bleu_score,
    compute_character_error_rate,
    compute_chrf_score,
    compute_exact_match,
    compute_expected_calibration_error,
    compute_ner_metrics,
    compute_relation_metrics,
    compute_seq2seq_metrics,
)
from rosetta.evaluation.validators import (
    ErrorType,
    ValidationResult,
    categorize_ner_errors,
    categorize_relation_errors,
    compute_historian_utility_score,
    compute_prediction_uncertainty,
    estimate_task_completion_time,
    validate_annotated_document,
    validate_entity_spans,
    validate_relations,
)

__all__ = [
    # Metrics
    "compute_ner_metrics",
    "compute_relation_metrics",
    "compute_bleu_score",
    "compute_chrf_score",
    "compute_exact_match",
    "compute_character_error_rate",
    "compute_expected_calibration_error",
    "compute_seq2seq_metrics",
    "bootstrap_confidence_interval",
    # Validators
    "ValidationResult",
    "ErrorType",
    "validate_entity_spans",
    "validate_relations",
    "validate_annotated_document",
    "compute_prediction_uncertainty",
    "categorize_ner_errors",
    "categorize_relation_errors",
    "compute_historian_utility_score",
    "estimate_task_completion_time",
]
