"""Comprehensive evaluation metrics for the Rosetta Transformer project.

This module provides metrics for evaluating NER, relation extraction, and seq2seq
tasks on ancient texts, including calibration metrics and bootstrap confidence intervals.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from rosetta.data.schemas import Entity, EntityLabel, Relation, RelationLabel
from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


def compute_ner_metrics(
    predictions: List[List[Entity]],
    references: List[List[Entity]],
    labels: Optional[List[EntityLabel]] = None,
    average: str = "micro",
) -> Dict[str, float]:
    """Compute NER metrics: precision, recall, F1.

    Args:
        predictions: List of predicted entity lists for each document
        references: List of reference entity lists for each document
        labels: Optional list of entity labels to evaluate
        average: Averaging strategy ('micro', 'macro', 'weighted')

    Returns:
        Dictionary with precision, recall, F1 scores (overall and per-entity-type)
    """
    if not predictions or not references:
        logger.warning("Empty predictions or references")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    # Convert entities to (doc_id, start, end, label) tuples
    pred_tuples = []
    ref_tuples = []

    for doc_id, (pred_ents, ref_ents) in enumerate(zip(predictions, references)):
        pred_tuples.extend([(doc_id, e.start, e.end, e.label) for e in pred_ents])
        ref_tuples.extend([(doc_id, e.start, e.end, e.label) for e in ref_ents])

    # Convert to sets for exact match
    pred_set = set(pred_tuples)
    ref_set = set(ref_tuples)

    # True positives: exact match (span and label)
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)

    # Overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(ref_tuples),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }

    # Per-entity-type metrics
    if labels is None:
        labels = list(set([e[3] for e in ref_tuples]))

    per_type_metrics = {}
    for label in labels:
        pred_label = {t for t in pred_set if t[3] == label}
        ref_label = {t for t in ref_set if t[3] == label}

        tp_label = len(pred_label & ref_label)
        fp_label = len(pred_label - ref_label)
        fn_label = len(ref_label - pred_label)

        prec = tp_label / (tp_label + fp_label) if (tp_label + fp_label) > 0 else 0.0
        rec = tp_label / (tp_label + fn_label) if (tp_label + fn_label) > 0 else 0.0
        f1_label = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_type_metrics[f"{label}_precision"] = prec
        per_type_metrics[f"{label}_recall"] = rec
        per_type_metrics[f"{label}_f1"] = f1_label
        per_type_metrics[f"{label}_support"] = len(ref_label)

    metrics.update(per_type_metrics)

    # Compute macro/weighted averages
    if average in ["macro", "weighted"]:
        label_f1s = [metrics[f"{label}_f1"] for label in labels]
        label_supports = [metrics[f"{label}_support"] for label in labels]

        if average == "macro":
            metrics["macro_f1"] = np.mean(label_f1s)
        elif average == "weighted" and sum(label_supports) > 0:
            metrics["weighted_f1"] = np.average(label_f1s, weights=label_supports)

    return metrics


def compute_relation_metrics(
    predictions: List[List[Relation]],
    references: List[List[Relation]],
    labels: Optional[List[RelationLabel]] = None,
) -> Dict[str, float]:
    """Compute relation extraction metrics: precision, recall, F1.

    Args:
        predictions: List of predicted relation lists for each document
        references: List of reference relation lists for each document
        labels: Optional list of relation labels to evaluate

    Returns:
        Dictionary with precision, recall, F1 scores
    """
    if not predictions or not references:
        logger.warning("Empty predictions or references")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    # Convert relations to tuples (doc_id, head_span, tail_span, label)
    pred_tuples = []
    ref_tuples = []

    for doc_id, (pred_rels, ref_rels) in enumerate(zip(predictions, references)):
        pred_tuples.extend(
            [
                (
                    doc_id,
                    (r.head.start, r.head.end),
                    (r.tail.start, r.tail.end),
                    r.label,
                )
                for r in pred_rels
            ]
        )
        ref_tuples.extend(
            [
                (
                    doc_id,
                    (r.head.start, r.head.end),
                    (r.tail.start, r.tail.end),
                    r.label,
                )
                for r in ref_rels
            ]
        )

    # Convert to sets for exact match
    pred_set = set(pred_tuples)
    ref_set = set(ref_tuples)

    # True positives: exact match
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)

    # Overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(ref_tuples),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }

    # Per-relation-type metrics
    if labels is None:
        labels = list(set([r[3] for r in ref_tuples]))

    for label in labels:
        pred_label = {t for t in pred_set if t[3] == label}
        ref_label = {t for t in ref_set if t[3] == label}

        tp_label = len(pred_label & ref_label)
        fp_label = len(pred_label - ref_label)
        fn_label = len(ref_label - pred_label)

        prec = tp_label / (tp_label + fp_label) if (tp_label + fp_label) > 0 else 0.0
        rec = tp_label / (tp_label + fn_label) if (tp_label + fn_label) > 0 else 0.0
        f1_label = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics[f"{label}_precision"] = prec
        metrics[f"{label}_recall"] = rec
        metrics[f"{label}_f1"] = f1_label
        metrics[f"{label}_support"] = len(ref_label)

    return metrics


def compute_bleu_score(
    predictions: List[str],
    references: List[Union[str, List[str]]],
    max_order: int = 4,
) -> Dict[str, float]:
    """Compute BLEU score for translation/transliteration.

    Args:
        predictions: List of predicted texts
        references: List of reference texts (can be list of lists for multiple refs)
        max_order: Maximum n-gram order (default: 4 for BLEU-4)

    Returns:
        Dictionary with BLEU score and n-gram precisions
    """
    try:
        from sacrebleu import corpus_bleu
    except ImportError:
        logger.warning("sacrebleu not installed, computing simple BLEU")
        return _compute_simple_bleu(predictions, references, max_order)

    # Convert to sacrebleu format
    if isinstance(references[0], str):
        references = [[ref] for ref in references]
    elif isinstance(references[0], list):
        # Transpose list of lists
        references = list(zip(*references))

    bleu = corpus_bleu(predictions, references, max_order=max_order)

    return {
        "bleu": bleu.score,
        "bleu_1": bleu.precisions[0] if len(bleu.precisions) > 0 else 0.0,
        "bleu_2": bleu.precisions[1] if len(bleu.precisions) > 1 else 0.0,
        "bleu_3": bleu.precisions[2] if len(bleu.precisions) > 2 else 0.0,
        "bleu_4": bleu.precisions[3] if len(bleu.precisions) > 3 else 0.0,
    }


def _compute_simple_bleu(
    predictions: List[str],
    references: List[Union[str, List[str]]],
    max_order: int = 4,
) -> Dict[str, float]:
    """Simple BLEU implementation when sacrebleu is not available."""

    def get_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams[ngram] += 1
        return ngrams

    total_precisions = [0.0] * max_order
    total_matches = [0] * max_order
    total_possible = [0] * max_order

    for pred, ref in zip(predictions, references):
        if isinstance(ref, str):
            ref = [ref]

        pred_tokens = pred.split()
        ref_tokens_list = [r.split() for r in ref]

        for n in range(1, max_order + 1):
            pred_ngrams = get_ngrams(pred_tokens, n)
            max_ref_ngrams = defaultdict(int)

            for ref_tokens in ref_tokens_list:
                ref_ngrams = get_ngrams(ref_tokens, n)
                for ngram in ref_ngrams:
                    max_ref_ngrams[ngram] = max(
                        max_ref_ngrams[ngram], ref_ngrams[ngram]
                    )

            matches = sum(
                min(pred_ngrams[ng], max_ref_ngrams[ng]) for ng in pred_ngrams
            )
            possible = max(len(pred_tokens) - n + 1, 0)

            total_matches[n - 1] += matches
            total_possible[n - 1] += possible

    precisions = [
        m / p if p > 0 else 0.0 for m, p in zip(total_matches, total_possible)
    ]

    # Compute geometric mean
    bleu = np.exp(np.mean([np.log(p) if p > 0 else -np.inf for p in precisions]))
    if not np.isfinite(bleu):
        bleu = 0.0

    return {
        "bleu": bleu * 100,
        "bleu_1": precisions[0] * 100 if len(precisions) > 0 else 0.0,
        "bleu_2": precisions[1] * 100 if len(precisions) > 1 else 0.0,
        "bleu_3": precisions[2] * 100 if len(precisions) > 2 else 0.0,
        "bleu_4": precisions[3] * 100 if len(precisions) > 3 else 0.0,
    }


def compute_chrf_score(
    predictions: List[str],
    references: List[Union[str, List[str]]],
    char_order: int = 6,
    word_order: int = 0,
    beta: float = 2.0,
) -> float:
    """Compute chrF score (character n-gram F-score).

    Args:
        predictions: List of predicted texts
        references: List of reference texts
        char_order: Maximum character n-gram order
        word_order: Maximum word n-gram order
        beta: Beta parameter for F-score

    Returns:
        chrF score
    """
    try:
        from sacrebleu import corpus_chrf
    except ImportError:
        logger.warning("sacrebleu not installed, returning 0.0 for chrF")
        return 0.0

    # Convert to sacrebleu format
    if isinstance(references[0], str):
        references = [[ref] for ref in references]
    elif isinstance(references[0], list):
        references = list(zip(*references))

    chrf = corpus_chrf(
        predictions,
        references,
        char_order=char_order,
        word_order=word_order,
        beta=beta,
    )

    return chrf.score


def compute_exact_match(
    predictions: List[str],
    references: List[str],
) -> float:
    """Compute exact match accuracy.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        Exact match percentage
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return 100 * matches / len(predictions) if predictions else 0.0


def compute_character_error_rate(
    predictions: List[str],
    references: List[str],
) -> float:
    """Compute character error rate (CER).

    CER is the Levenshtein distance normalized by reference length.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        Average character error rate
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    def levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    total_cer = 0.0
    for pred, ref in zip(predictions, references):
        if len(ref) == 0:
            total_cer += 1.0 if len(pred) > 0 else 0.0
        else:
            distance = levenshtein_distance(pred, ref)
            total_cer += distance / len(ref)

    return total_cer / len(predictions) if predictions else 0.0


def compute_expected_calibration_error(
    predictions: np.ndarray,
    confidences: np.ndarray,
    references: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE).

    ECE measures the difference between confidence and accuracy.

    Args:
        predictions: Predicted labels (N,)
        confidences: Confidence scores (N,)
        references: True labels (N,)
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with ECE and calibration data for reliability diagrams
    """
    if len(predictions) != len(confidences) or len(predictions) != len(references):
        raise ValueError(
            "predictions, confidences, and references must have same length"
        )

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = predictions == references
    ece = 0.0

    bin_data = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bin_data.append(
                {
                    "confidence": float(avg_confidence_in_bin),
                    "accuracy": float(accuracy_in_bin),
                    "count": int(in_bin.sum()),
                }
            )
        else:
            bin_data.append(
                {
                    "confidence": float((bin_lower + bin_upper) / 2),
                    "accuracy": 0.0,
                    "count": 0,
                }
            )

    return {
        "ece": float(ece),
        "n_bins": n_bins,
        "bins": bin_data,
    }


def bootstrap_confidence_interval(
    metric_fn: Callable,
    predictions: Any,
    references: Any,
    n_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence intervals for a metric.

    Args:
        metric_fn: Function that computes metric from (predictions, references)
        predictions: Predictions (list or array)
        references: References (list or array)
        n_samples: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with mean, std, and confidence interval bounds
    """
    np.random.seed(seed)
    n = len(predictions)

    if n == 0:
        return {"mean": 0.0, "std": 0.0, "lower": 0.0, "upper": 0.0}

    bootstrap_scores = []

    for _ in range(n_samples):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)

        # Handle different data types
        if isinstance(predictions, (list, tuple)):
            boot_pred = [predictions[i] for i in indices]
            boot_ref = [references[i] for i in indices]
        else:  # numpy array
            boot_pred = predictions[indices]
            boot_ref = references[indices]

        # Compute metric
        try:
            score = metric_fn(boot_pred, boot_ref)
            # Handle dict returns (take first value)
            if isinstance(score, dict):
                score = list(score.values())[0]
            bootstrap_scores.append(score)
        except Exception as e:
            logger.warning(f"Bootstrap sample failed: {e}")
            continue

    if not bootstrap_scores:
        return {"mean": 0.0, "std": 0.0, "lower": 0.0, "upper": 0.0}

    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    return {
        "mean": float(np.mean(bootstrap_scores)),
        "std": float(np.std(bootstrap_scores)),
        "lower": float(np.percentile(bootstrap_scores, lower_percentile)),
        "upper": float(np.percentile(bootstrap_scores, upper_percentile)),
        "n_samples": n_samples,
        "confidence_level": confidence_level,
    }


def compute_seq2seq_metrics(
    predictions: List[str],
    references: List[Union[str, List[str]]],
    include_bootstrap: bool = True,
) -> Dict[str, Any]:
    """Compute comprehensive seq2seq metrics.

    Args:
        predictions: List of predicted texts
        references: List of reference texts
        include_bootstrap: Whether to include bootstrap confidence intervals

    Returns:
        Dictionary with all seq2seq metrics
    """
    # Ensure references is in correct format
    if isinstance(references[0], str):
        single_refs = references
    else:
        single_refs = [r[0] if isinstance(r, list) else r for r in references]

    metrics = {}

    # BLEU
    bleu_metrics = compute_bleu_score(predictions, references)
    metrics.update(bleu_metrics)

    # chrF
    metrics["chrf"] = compute_chrf_score(predictions, references)

    # Exact match
    metrics["exact_match"] = compute_exact_match(predictions, single_refs)

    # Character error rate
    metrics["cer"] = compute_character_error_rate(predictions, single_refs)

    # Bootstrap confidence intervals
    if include_bootstrap:
        logger.info("Computing bootstrap confidence intervals...")

        # BLEU CI
        bleu_ci = bootstrap_confidence_interval(
            lambda p, r: compute_bleu_score(p, r)["bleu"],
            predictions,
            references,
        )
        metrics["bleu_ci"] = bleu_ci

        # Exact match CI
        em_ci = bootstrap_confidence_interval(
            compute_exact_match,
            predictions,
            single_refs,
        )
        metrics["exact_match_ci"] = em_ci

    return metrics
