"""Prediction validation and quality assessment for Rosetta Transformer.

This module provides functions for validating predictions, checking consistency,
computing quality scores, and categorizing errors for model outputs.
"""

from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np

from rosetta.data.schemas import (
    AnnotatedDocument,
    Entity,
    EntityLabel,
    Relation,
)
from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


class ErrorType(str, Enum):
    """Types of prediction errors."""

    # Entity errors
    SPAN_BOUNDARY = "span_boundary"  # Wrong entity boundaries
    ENTITY_TYPE = "entity_type"  # Wrong entity type
    MISSING_ENTITY = "missing_entity"  # False negative
    SPURIOUS_ENTITY = "spurious_entity"  # False positive
    OVERLAPPING_ENTITIES = "overlapping_entities"  # Overlapping spans

    # Relation errors
    WRONG_RELATION_TYPE = "wrong_relation_type"
    MISSING_RELATION = "missing_relation"
    SPURIOUS_RELATION = "spurious_relation"
    INVALID_ENTITY_PAIR = "invalid_entity_pair"

    # Consistency errors
    SPAN_OUT_OF_BOUNDS = "span_out_of_bounds"
    EMPTY_SPAN = "empty_span"
    RELATION_WITHOUT_ENTITIES = "relation_without_entities"

    # Seq2seq errors
    EMPTY_OUTPUT = "empty_output"
    LENGTH_MISMATCH = "length_mismatch"
    CHARACTER_ERROR = "character_error"


class ValidationResult:
    """Result of prediction validation."""

    def __init__(
        self,
        is_valid: bool = True,
        errors: Optional[List[Dict[str, Any]]] = None,
        warnings: Optional[List[str]] = None,
        quality_score: float = 1.0,
    ):
        """Initialize validation result.

        Args:
            is_valid: Whether prediction is valid
            errors: List of error dictionaries
            warnings: List of warning messages
            quality_score: Quality score from 0.0 to 1.0
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.quality_score = quality_score

    def add_error(self, error_type: ErrorType, message: str, **kwargs) -> None:
        """Add an error to the validation result.

        Args:
            error_type: Type of error
            message: Error message
            **kwargs: Additional error metadata
        """
        self.is_valid = False
        self.errors.append(
            {
                "type": error_type.value,
                "message": message,
                **kwargs,
            }
        )

    def add_warning(self, message: str) -> None:
        """Add a warning to the validation result.

        Args:
            message: Warning message
        """
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "quality_score": self.quality_score,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


def validate_entity_spans(
    entities: List[Entity],
    text: str,
    allow_overlapping: bool = False,
) -> ValidationResult:
    """Validate entity spans for consistency.

    Args:
        entities: List of entities to validate
        text: Document text
        allow_overlapping: Whether to allow overlapping entities

    Returns:
        ValidationResult with validation status and errors
    """
    result = ValidationResult()
    text_len = len(text)

    # Check each entity
    for i, entity in enumerate(entities):
        # Check span bounds
        if entity.start < 0 or entity.end > text_len:
            result.add_error(
                ErrorType.SPAN_OUT_OF_BOUNDS,
                f"Entity span [{entity.start}:{entity.end}] out of bounds for text length {text_len}",
                entity_index=i,
                entity=entity.text,
            )
            continue

        # Check empty span
        if entity.start >= entity.end:
            result.add_error(
                ErrorType.EMPTY_SPAN,
                f"Empty or invalid span [{entity.start}:{entity.end}]",
                entity_index=i,
            )
            continue

        # Check span matches text
        actual_text = text[entity.start : entity.end]
        if actual_text != entity.text:
            result.add_warning(
                f"Entity text '{entity.text}' does not match document span '{actual_text}'"
            )

    # Check overlapping entities
    if not allow_overlapping and len(entities) > 1:
        sorted_entities = sorted(entities, key=lambda e: e.start)
        for i in range(len(sorted_entities) - 1):
            if sorted_entities[i].end > sorted_entities[i + 1].start:
                result.add_error(
                    ErrorType.OVERLAPPING_ENTITIES,
                    f"Overlapping entities: '{sorted_entities[i].text}' and '{sorted_entities[i + 1].text}'",
                    entity1_span=(sorted_entities[i].start, sorted_entities[i].end),
                    entity2_span=(
                        sorted_entities[i + 1].start,
                        sorted_entities[i + 1].end,
                    ),
                )

    # Compute quality score
    if entities:
        error_penalty = len(result.errors) / len(entities)
        result.quality_score = max(0.0, 1.0 - error_penalty)

    return result


def validate_relations(
    relations: List[Relation],
    entities: List[Entity],
) -> ValidationResult:
    """Validate relations for consistency with entities.

    Args:
        relations: List of relations to validate
        entities: List of entities in the document

    Returns:
        ValidationResult with validation status and errors
    """
    result = ValidationResult()

    # Create entity span set for lookup
    entity_spans = {(e.start, e.end, e.label) for e in entities}

    for i, relation in enumerate(relations):
        # Check head entity exists
        head_span = (relation.head.start, relation.head.end, relation.head.label)
        if head_span not in entity_spans:
            result.add_error(
                ErrorType.INVALID_ENTITY_PAIR,
                f"Head entity not found in entity list: {relation.head.text}",
                relation_index=i,
                head_span=head_span,
            )

        # Check tail entity exists
        tail_span = (relation.tail.start, relation.tail.end, relation.tail.label)
        if tail_span not in entity_spans:
            result.add_error(
                ErrorType.INVALID_ENTITY_PAIR,
                f"Tail entity not found in entity list: {relation.tail.text}",
                relation_index=i,
                tail_span=tail_span,
            )

        # Check head and tail are different
        if (
            relation.head.start == relation.tail.start
            and relation.head.end == relation.tail.end
        ):
            result.add_error(
                ErrorType.INVALID_ENTITY_PAIR,
                "Head and tail entities are the same",
                relation_index=i,
            )

    # Compute quality score
    if relations:
        error_penalty = len(result.errors) / len(relations)
        result.quality_score = max(0.0, 1.0 - error_penalty)

    return result


def validate_annotated_document(
    document: AnnotatedDocument,
    allow_overlapping_entities: bool = False,
) -> ValidationResult:
    """Validate entire annotated document for consistency.

    Args:
        document: Annotated document to validate
        allow_overlapping_entities: Whether to allow overlapping entities

    Returns:
        ValidationResult with validation status and errors
    """
    result = ValidationResult()

    # Validate entities
    entity_result = validate_entity_spans(
        document.entities,
        document.text,
        allow_overlapping=allow_overlapping_entities,
    )
    result.errors.extend(entity_result.errors)
    result.warnings.extend(entity_result.warnings)

    # Validate relations
    if document.relations:
        relation_result = validate_relations(document.relations, document.entities)
        result.errors.extend(relation_result.errors)
        result.warnings.extend(relation_result.warnings)

    # Update validity and quality score
    result.is_valid = len(result.errors) == 0
    total_annotations = len(document.entities) + len(document.relations)
    if total_annotations > 0:
        error_penalty = len(result.errors) / total_annotations
        result.quality_score = max(0.0, 1.0 - error_penalty)

    return result


def compute_prediction_uncertainty(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    method: str = "entropy",
) -> np.ndarray:
    """Compute uncertainty estimates for predictions.

    Args:
        predictions: Predicted labels (N,)
        probabilities: Prediction probabilities (N, num_classes)
        method: Uncertainty estimation method ('entropy', 'margin', 'confidence')

    Returns:
        Uncertainty scores (N,)
    """
    if method == "entropy":
        # Shannon entropy: -sum(p * log(p))
        epsilon = 1e-10
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=-1)
        # Normalize to [0, 1]
        max_entropy = np.log(probabilities.shape[-1])
        return entropy / max_entropy

    elif method == "margin":
        # Margin: difference between top two probabilities
        sorted_probs = np.sort(probabilities, axis=-1)
        if probabilities.shape[-1] > 1:
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            # Return 1 - margin (higher margin = lower uncertainty)
            return 1 - margin
        else:
            return np.zeros(len(predictions))

    elif method == "confidence":
        # Confidence: 1 - max probability
        max_probs = np.max(probabilities, axis=-1)
        return 1 - max_probs

    else:
        raise ValueError(f"Unknown uncertainty method: {method}")


def categorize_ner_errors(
    predictions: List[List[Entity]],
    references: List[List[Entity]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize NER prediction errors by type.

    Args:
        predictions: List of predicted entity lists
        references: List of reference entity lists

    Returns:
        Dictionary mapping error types to lists of error instances
    """
    errors = defaultdict(list)

    for doc_id, (pred_ents, ref_ents) in enumerate(zip(predictions, references)):
        # Create span mappings
        pred_spans = {(e.start, e.end): e for e in pred_ents}
        ref_spans = {(e.start, e.end): e for e in ref_ents}

        pred_span_labels = {(e.start, e.end, e.label) for e in pred_ents}
        ref_span_labels = {(e.start, e.end, e.label) for e in ref_ents}

        # Missing entities (false negatives)
        for ref_ent in ref_ents:
            ref_key = (ref_ent.start, ref_ent.end, ref_ent.label)
            if ref_key not in pred_span_labels:
                # Check if span exists with different label
                span_key = (ref_ent.start, ref_ent.end)
                if span_key in pred_spans:
                    errors[ErrorType.ENTITY_TYPE].append(
                        {
                            "doc_id": doc_id,
                            "span": span_key,
                            "text": ref_ent.text,
                            "predicted_label": pred_spans[span_key].label,
                            "true_label": ref_ent.label,
                        }
                    )
                else:
                    errors[ErrorType.MISSING_ENTITY].append(
                        {
                            "doc_id": doc_id,
                            "span": span_key,
                            "text": ref_ent.text,
                            "label": ref_ent.label,
                        }
                    )

        # Spurious entities (false positives)
        for pred_ent in pred_ents:
            pred_key = (pred_ent.start, pred_ent.end, pred_ent.label)
            if pred_key not in ref_span_labels:
                span_key = (pred_ent.start, pred_ent.end)
                # Don't double-count entity type errors
                if span_key not in ref_spans:
                    errors[ErrorType.SPURIOUS_ENTITY].append(
                        {
                            "doc_id": doc_id,
                            "span": span_key,
                            "text": pred_ent.text,
                            "label": pred_ent.label,
                        }
                    )

        # Check for span boundary errors (overlapping but not exact match)
        for pred_ent in pred_ents:
            for ref_ent in ref_ents:
                if pred_ent.label == ref_ent.label:
                    # Check if spans overlap but are not exact
                    overlap = max(pred_ent.start, ref_ent.start) < min(
                        pred_ent.end, ref_ent.end
                    )
                    exact_match = (
                        pred_ent.start == ref_ent.start and pred_ent.end == ref_ent.end
                    )
                    if overlap and not exact_match:
                        errors[ErrorType.SPAN_BOUNDARY].append(
                            {
                                "doc_id": doc_id,
                                "predicted_span": (pred_ent.start, pred_ent.end),
                                "true_span": (ref_ent.start, ref_ent.end),
                                "predicted_text": pred_ent.text,
                                "true_text": ref_ent.text,
                                "label": pred_ent.label,
                            }
                        )

    return dict(errors)  # type: ignore[arg-type,return-value]


def categorize_relation_errors(
    predictions: List[List[Relation]],
    references: List[List[Relation]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize relation extraction errors by type.

    Args:
        predictions: List of predicted relation lists
        references: List of reference relation lists

    Returns:
        Dictionary mapping error types to lists of error instances
    """
    errors = defaultdict(list)

    for doc_id, (pred_rels, ref_rels) in enumerate(zip(predictions, references)):
        # Create relation mappings
        pred_rel_keys = {
            ((r.head.start, r.head.end), (r.tail.start, r.tail.end), r.label): r
            for r in pred_rels
        }
        ref_rel_keys = {
            ((r.head.start, r.head.end), (r.tail.start, r.tail.end), r.label): r
            for r in ref_rels
        }

        pred_entity_pairs = {
            ((r.head.start, r.head.end), (r.tail.start, r.tail.end)): r
            for r in pred_rels
        }
        ref_entity_pairs = {
            ((r.head.start, r.head.end), (r.tail.start, r.tail.end)): r
            for r in ref_rels
        }

        # Missing relations (false negatives)
        for ref_key, ref_rel in ref_rel_keys.items():
            if ref_key not in pred_rel_keys:
                entity_pair = ref_key[:2]
                # Check if entity pair exists with different label
                if entity_pair in pred_entity_pairs:
                    errors[ErrorType.WRONG_RELATION_TYPE].append(
                        {
                            "doc_id": doc_id,
                            "head_span": ref_key[0],
                            "tail_span": ref_key[1],
                            "predicted_label": pred_entity_pairs[entity_pair].label,
                            "true_label": ref_rel.label,
                        }
                    )
                else:
                    errors[ErrorType.MISSING_RELATION].append(
                        {
                            "doc_id": doc_id,
                            "head_span": ref_key[0],
                            "tail_span": ref_key[1],
                            "label": ref_rel.label,
                        }
                    )

        # Spurious relations (false positives)
        for pred_key, pred_rel in pred_rel_keys.items():
            if pred_key not in ref_rel_keys:
                entity_pair = pred_key[:2]
                # Don't double-count wrong relation type errors
                if entity_pair not in ref_entity_pairs:
                    errors[ErrorType.SPURIOUS_RELATION].append(
                        {
                            "doc_id": doc_id,
                            "head_span": pred_key[0],
                            "tail_span": pred_key[1],
                            "label": pred_rel.label,
                        }
                    )

    return dict(errors)  # type: ignore[arg-type,return-value]


def compute_historian_utility_score(
    predictions: List[List[Entity]],
    references: List[List[Entity]],
    high_value_entities: Optional[Set[EntityLabel]] = None,
) -> Dict[str, float]:
    """Compute utility scores for historian use cases.

    This metric emphasizes the practical value of predictions for historians,
    prioritizing high-value entity types and penalizing spurious predictions.

    Args:
        predictions: List of predicted entity lists
        references: List of reference entity lists
        high_value_entities: Set of high-value entity labels (e.g., PERSON, LOCATION)

    Returns:
        Dictionary with utility scores
    """
    if high_value_entities is None:
        high_value_entities = {
            EntityLabel.PERSON,
            EntityLabel.LOCATION,
            EntityLabel.DATE,
            EntityLabel.EVENT,
            EntityLabel.DEITY,
        }

    total_high_value_correct = 0
    total_high_value_ref = 0
    total_high_value_pred = 0
    total_correct = 0
    total_ref = 0
    total_spurious = 0

    for pred_ents, ref_ents in zip(predictions, references):
        pred_set = {(e.start, e.end, e.label) for e in pred_ents}
        ref_set = {(e.start, e.end, e.label) for e in ref_ents}

        # Overall counts
        correct = pred_set & ref_set
        total_correct += len(correct)
        total_ref += len(ref_set)
        total_spurious += len(pred_set - ref_set)

        # High-value entity counts
        high_value_pred = {t for t in pred_set if t[2] in high_value_entities}
        high_value_ref = {t for t in ref_set if t[2] in high_value_entities}
        high_value_correct = high_value_pred & high_value_ref

        total_high_value_correct += len(high_value_correct)
        total_high_value_ref += len(high_value_ref)
        total_high_value_pred += len(high_value_pred)

    # Compute scores
    recall = total_correct / total_ref if total_ref > 0 else 0.0
    precision = (
        total_correct / (total_correct + total_spurious)
        if (total_correct + total_spurious) > 0
        else 0.0
    )

    high_value_recall = (
        total_high_value_correct / total_high_value_ref
        if total_high_value_ref > 0
        else 0.0
    )
    high_value_precision = (
        total_high_value_correct / total_high_value_pred
        if total_high_value_pred > 0
        else 0.0
    )

    # Utility score: weighted combination favoring high-value entities
    # and penalizing spurious predictions more heavily
    utility_score = (
        0.4 * high_value_recall
        + 0.3 * high_value_precision
        + 0.2 * recall
        + 0.1 * precision
    )

    return {
        "utility_score": utility_score,
        "high_value_recall": high_value_recall,
        "high_value_precision": high_value_precision,
        "overall_recall": recall,
        "overall_precision": precision,
        "spurious_rate": total_spurious / total_ref if total_ref > 0 else 0.0,
    }


def estimate_task_completion_time(
    predictions: List[List[Entity]],
    references: List[List[Entity]],
    base_time_per_doc: float = 300.0,  # 5 minutes per document
    correction_time_per_error: float = 10.0,  # 10 seconds per error
) -> Dict[str, float]:
    """Estimate time savings for historians using model predictions.

    Args:
        predictions: List of predicted entity lists
        references: List of reference entity lists
        base_time_per_doc: Base annotation time per document (seconds)
        correction_time_per_error: Time to correct one error (seconds)

    Returns:
        Dictionary with time estimates
    """
    n_docs = len(predictions)
    total_errors = 0

    for pred_ents, ref_ents in zip(predictions, references):
        pred_set = {(e.start, e.end, e.label) for e in pred_ents}
        ref_set = {(e.start, e.end, e.label) for e in ref_ents}

        # Count errors (both false positives and false negatives)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)
        total_errors += fp + fn

    # Time estimates
    baseline_time = n_docs * base_time_per_doc
    correction_time = total_errors * correction_time_per_error
    total_time_with_model = correction_time

    time_saved = baseline_time - total_time_with_model
    time_saved_percentage = (
        (time_saved / baseline_time * 100) if baseline_time > 0 else 0.0
    )

    return {
        "baseline_time_seconds": baseline_time,
        "correction_time_seconds": correction_time,
        "total_time_with_model_seconds": total_time_with_model,
        "time_saved_seconds": time_saved,
        "time_saved_percentage": time_saved_percentage,
        "average_errors_per_doc": total_errors / n_docs if n_docs > 0 else 0.0,
    }
