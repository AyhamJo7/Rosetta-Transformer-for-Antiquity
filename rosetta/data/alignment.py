"""Parallel text alignment for ancient texts.

This module provides classes for aligning parallel texts in different languages
or scripts, computing alignment quality scores, and building parallel corpora
for translation tasks.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import jsonlines
import numpy as np
from tqdm import tqdm

from rosetta.data.schemas import Document, ParallelText
from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


class TextAligner:
    """Align parallel texts in different languages or scripts.

    Supports alignment of ancient texts with their transliterations
    or translations, using various alignment methods including
    length-based, lexical, and sentence-level alignment.
    """

    def __init__(
        self,
        method: str = "sentence",
        length_ratio_threshold: float = 3.0,
        min_alignment_score: float = 0.3,
    ):
        """Initialize the text aligner.

        Args:
            method: Alignment method ('sentence', 'length', 'lexical')
            length_ratio_threshold: Maximum acceptable length ratio between texts
            min_alignment_score: Minimum score for accepting an alignment

        Raises:
            ValueError: If method is not supported
        """
        if method not in ["sentence", "length", "lexical"]:
            raise ValueError(f"Unsupported alignment method: {method}")

        self.method = method
        self.length_ratio_threshold = length_ratio_threshold
        self.min_alignment_score = min_alignment_score
        self.logger = get_logger(__name__)

    def _compute_length_score(self, source_text: str, target_text: str) -> float:
        """Compute alignment score based on text lengths.

        Args:
            source_text: Source language text
            target_text: Target language text

        Returns:
            Alignment score from 0.0 to 1.0
        """
        source_len = len(source_text)
        target_len = len(target_text)

        if source_len == 0 or target_len == 0:
            return 0.0

        # Calculate length ratio
        ratio = max(source_len, target_len) / min(source_len, target_len)

        # Convert ratio to score (lower ratio = higher score)
        # Ratio of 1.0 -> score of 1.0
        # Ratio >= threshold -> score of 0.0
        if ratio >= self.length_ratio_threshold:
            return 0.0

        score = 1.0 - ((ratio - 1.0) / (self.length_ratio_threshold - 1.0))
        return max(0.0, min(1.0, score))

    def _compute_lexical_score(self, source_text: str, target_text: str) -> float:
        """Compute alignment score based on lexical overlap.

        Uses character n-gram overlap for cross-script matching.

        Args:
            source_text: Source language text
            target_text: Target language text

        Returns:
            Alignment score from 0.0 to 1.0
        """

        # Extract character bigrams
        def get_bigrams(text: str) -> set:
            text = text.lower()
            return set(text[i : i + 2] for i in range(len(text) - 1))

        source_bigrams = get_bigrams(source_text)
        target_bigrams = get_bigrams(target_text)

        if not source_bigrams or not target_bigrams:
            return 0.0

        # Calculate Jaccard similarity
        intersection = source_bigrams.intersection(target_bigrams)
        union = source_bigrams.union(target_bigrams)

        return len(intersection) / len(union) if union else 0.0

    def _align_sentences(
        self,
        source_sentences: List[str],
        target_sentences: List[str],
    ) -> List[Tuple[str, str, float]]:
        """Align parallel sentences.

        Simple 1-1 alignment assuming sentences are already aligned.

        Args:
            source_sentences: List of source language sentences
            target_sentences: List of target language sentences

        Returns:
            List of (source_sent, target_sent, score) tuples
        """
        alignments = []
        min_len = min(len(source_sentences), len(target_sentences))

        for i in range(min_len):
            source_sent = source_sentences[i]
            target_sent = target_sentences[i]

            # Compute alignment score
            length_score = self._compute_length_score(source_sent, target_sent)
            lexical_score = self._compute_lexical_score(source_sent, target_sent)

            # Combine scores (weighted average)
            score = 0.6 * length_score + 0.4 * lexical_score

            if score >= self.min_alignment_score:
                alignments.append((source_sent, target_sent, score))

        return alignments

    def align_documents(
        self,
        source_doc: Document,
        target_doc: Document,
        source_sentences: Optional[List[str]] = None,
        target_sentences: Optional[List[str]] = None,
    ) -> List[ParallelText]:
        """Align two documents.

        Args:
            source_doc: Source language document
            target_doc: Target language document
            source_sentences: Pre-segmented source sentences (optional)
            target_sentences: Pre-segmented target sentences (optional)

        Returns:
            List of ParallelText objects with alignments
        """
        if self.method == "sentence":
            if source_sentences is None or target_sentences is None:
                raise ValueError("Sentence method requires pre-segmented sentences")

            alignments = self._align_sentences(source_sentences, target_sentences)

            parallel_texts = []
            for idx, (src, tgt, score) in enumerate(alignments):
                parallel_texts.append(
                    ParallelText(  # type: ignore[call-arg]
                        source_text=src,
                        target_text=tgt,
                        alignment_score=score,
                        language_pair=(source_doc.language, target_doc.language),
                        source_id=f"{source_doc.id}_sent_{idx}",
                        target_id=f"{target_doc.id}_sent_{idx}",
                        metadata={
                            "alignment_method": self.method,
                            "source_doc_id": source_doc.id,
                            "target_doc_id": target_doc.id,
                        },
                    )
                )

            return parallel_texts

        else:  # Document-level alignment
            if self.method == "length":
                score = self._compute_length_score(source_doc.text, target_doc.text)
            else:  # lexical
                score = self._compute_lexical_score(source_doc.text, target_doc.text)

            if score >= self.min_alignment_score:
                return [
                    ParallelText(  # type: ignore[call-arg]
                        source_text=source_doc.text,
                        target_text=target_doc.text,
                        alignment_score=score,
                        language_pair=(source_doc.language, target_doc.language),
                        source_id=source_doc.id,
                        target_id=target_doc.id,
                        metadata={
                            "alignment_method": self.method,
                        },
                    )
                ]
            else:
                return []

    def align_corpus(
        self,
        source_docs: List[Document],
        target_docs: List[Document],
        show_progress: bool = True,
    ) -> List[ParallelText]:
        """Align two parallel corpora.

        Assumes documents are already paired (same length and order).

        Args:
            source_docs: Source language documents
            target_docs: Target language documents
            show_progress: Whether to show progress bar

        Returns:
            List of ParallelText alignments

        Raises:
            ValueError: If document lists have different lengths
        """
        if len(source_docs) != len(target_docs):
            raise ValueError(
                f"Document lists must have same length: "
                f"{len(source_docs)} != {len(target_docs)}"
            )

        self.logger.info(
            f"Aligning {len(source_docs)} document pairs with {self.method} method"
        )

        parallel_texts = []

        doc_pairs = zip(source_docs, target_docs)
        if show_progress:
            doc_pairs = tqdm(list(doc_pairs), desc="Aligning")

        for source_doc, target_doc in doc_pairs:
            alignments = self.align_documents(source_doc, target_doc)
            parallel_texts.extend(alignments)

        self.logger.info(
            f"Alignment complete: {len(parallel_texts)} parallel text pairs created"
        )

        return parallel_texts


class AlignmentScorer:
    """Compute and evaluate alignment quality scores.

    Provides various metrics for assessing the quality of parallel text
    alignments, useful for filtering low-quality pairs.
    """

    def __init__(self):
        """Initialize the alignment scorer."""
        self.logger = get_logger(__name__)

    def length_ratio_score(self, parallel_text: ParallelText) -> float:
        """Compute score based on length ratio.

        Args:
            parallel_text: Parallel text pair

        Returns:
            Score from 0.0 to 1.0 (1.0 = perfect ratio)
        """
        source_len = len(parallel_text.source_text)
        target_len = len(parallel_text.target_text)

        if source_len == 0 or target_len == 0:
            return 0.0

        ratio = max(source_len, target_len) / min(source_len, target_len)

        # Ideal ratio is 1.0, penalize deviations
        # Ratio > 3.0 gets score near 0
        score = max(0.0, 1.0 - (ratio - 1.0) / 2.0)
        return score

    def word_count_ratio_score(self, parallel_text: ParallelText) -> float:
        """Compute score based on word count ratio.

        Args:
            parallel_text: Parallel text pair

        Returns:
            Score from 0.0 to 1.0
        """
        source_words = len(parallel_text.source_text.split())
        target_words = len(parallel_text.target_text.split())

        if source_words == 0 or target_words == 0:
            return 0.0

        ratio = max(source_words, target_words) / min(source_words, target_words)

        score = max(0.0, 1.0 - (ratio - 1.0) / 2.0)
        return score

    def compute_quality_score(self, parallel_text: ParallelText) -> float:
        """Compute overall quality score for an alignment.

        Combines multiple metrics into a single quality score.

        Args:
            parallel_text: Parallel text pair

        Returns:
            Overall quality score from 0.0 to 1.0
        """
        # Get individual scores
        length_score = self.length_ratio_score(parallel_text)
        word_score = self.word_count_ratio_score(parallel_text)
        alignment_score = parallel_text.alignment_score

        # Weighted average
        weights = [0.3, 0.3, 0.4]  # [length, word_count, alignment]
        scores = [length_score, word_score, alignment_score]

        quality_score = sum(w * s for w, s in zip(weights, scores))

        return quality_score

    def score_corpus(
        self,
        parallel_texts: List[ParallelText],
        show_progress: bool = True,
    ) -> List[Tuple[ParallelText, float]]:
        """Score a corpus of parallel texts.

        Args:
            parallel_texts: List of parallel text pairs
            show_progress: Whether to show progress bar

        Returns:
            List of (parallel_text, quality_score) tuples
        """
        self.logger.info(f"Scoring {len(parallel_texts)} parallel text pairs")

        text_iter = (
            tqdm(parallel_texts, desc="Scoring") if show_progress else parallel_texts
        )

        scored = [(pt, self.compute_quality_score(pt)) for pt in text_iter]

        avg_score = sum(score for _, score in scored) / len(scored) if scored else 0.0

        self.logger.info(f"Scoring complete: average quality score = {avg_score:.3f}")

        return scored


class ParallelCorpusBuilder:
    """Build parallel corpora with quality filtering.

    Combines alignment and scoring to create high-quality parallel
    corpora suitable for training translation models.
    """

    def __init__(
        self,
        aligner: Optional[TextAligner] = None,
        scorer: Optional[AlignmentScorer] = None,
        min_quality_score: float = 0.5,
        min_length: int = 3,
        max_length: int = 512,
    ):
        """Initialize the parallel corpus builder.

        Args:
            aligner: Text aligner instance (default: sentence aligner)
            scorer: Alignment scorer instance
            min_quality_score: Minimum quality score for inclusion
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters
        """
        self.aligner = aligner or TextAligner()
        self.scorer = scorer or AlignmentScorer()
        self.min_quality_score = min_quality_score
        self.min_length = min_length
        self.max_length = max_length
        self.logger = get_logger(__name__)

    def _filter_by_length(
        self, parallel_texts: List[ParallelText]
    ) -> List[ParallelText]:
        """Filter parallel texts by length constraints.

        Args:
            parallel_texts: List of parallel text pairs

        Returns:
            Filtered list
        """
        filtered = []

        for pt in parallel_texts:
            source_len = len(pt.source_text)
            target_len = len(pt.target_text)

            # Check both texts meet length requirements
            if (
                source_len >= self.min_length
                and target_len >= self.min_length
                and source_len <= self.max_length
                and target_len <= self.max_length
            ):
                filtered.append(pt)

        return filtered

    def _filter_by_quality(
        self,
        parallel_texts: List[ParallelText],
        show_progress: bool = True,
    ) -> List[ParallelText]:
        """Filter parallel texts by quality score.

        Args:
            parallel_texts: List of parallel text pairs
            show_progress: Whether to show progress bar

        Returns:
            Filtered list
        """
        scored = self.scorer.score_corpus(parallel_texts, show_progress)

        filtered = [pt for pt, score in scored if score >= self.min_quality_score]

        return filtered

    def build_corpus(
        self,
        source_docs: List[Document],
        target_docs: List[Document],
        output_file: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
    ) -> List[ParallelText]:
        """Build parallel corpus from document pairs.

        Args:
            source_docs: Source language documents
            target_docs: Target language documents
            output_file: Optional path to save corpus (JSONL format)
            show_progress: Whether to show progress bar

        Returns:
            List of filtered, high-quality parallel texts

        Note:
            After building corpus, consider versioning with DVC:
            $ dvc add {output_file}
        """
        self.logger.info(
            f"Building parallel corpus from {len(source_docs)} document pairs"
        )

        # Align documents
        parallel_texts = self.aligner.align_corpus(
            source_docs, target_docs, show_progress
        )

        initial_count = len(parallel_texts)
        self.logger.info(f"Initial alignments: {initial_count}")

        # Filter by length
        parallel_texts = self._filter_by_length(parallel_texts)
        after_length = len(parallel_texts)
        self.logger.info(
            f"After length filtering: {after_length} "
            f"(removed {initial_count - after_length})"
        )

        # Filter by quality
        parallel_texts = self._filter_by_quality(parallel_texts, show_progress)
        final_count = len(parallel_texts)
        self.logger.info(
            f"After quality filtering: {final_count} "
            f"(removed {after_length - final_count})"
        )

        # Save if output file specified
        if output_file:
            self.save_corpus(parallel_texts, output_file)

        return parallel_texts

    def save_corpus(
        self,
        parallel_texts: List[ParallelText],
        output_file: Union[str, Path],
        format: str = "jsonl",
    ) -> None:
        """Save parallel corpus to file.

        Args:
            parallel_texts: List of parallel text pairs
            output_file: Path to output file
            format: Output format ('jsonl' or 'json')

        Raises:
            ValueError: If format is not supported
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with jsonlines.open(output_file, mode="w") as writer:
                for pt in parallel_texts:
                    writer.write(pt.model_dump())
        elif format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    [pt.model_dump() for pt in parallel_texts],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Saved {len(parallel_texts)} parallel texts to {output_file}")
        self.logger.info(f"Consider versioning with DVC: dvc add {output_file}")

    def create_splits(
        self,
        parallel_texts: List[ParallelText],
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> Dict[str, List[ParallelText]]:
        """Create train/dev/test splits from parallel corpus.

        Args:
            parallel_texts: List of parallel text pairs
            train_ratio: Proportion for training set
            dev_ratio: Proportion for development set
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with 'train', 'dev', 'test' keys

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {train_ratio + dev_ratio + test_ratio}"
            )

        # Shuffle with fixed seed
        rng = np.random.default_rng(random_seed)
        indices = np.arange(len(parallel_texts))
        rng.shuffle(indices)

        shuffled = [parallel_texts[i] for i in indices]

        # Calculate split points
        n = len(shuffled)
        train_end = int(n * train_ratio)
        dev_end = train_end + int(n * dev_ratio)

        splits = {
            "train": shuffled[:train_end],
            "dev": shuffled[train_end:dev_end],
            "test": shuffled[dev_end:],
        }

        self.logger.info(
            f"Created splits: train={len(splits['train'])}, "
            f"dev={len(splits['dev'])}, test={len(splits['test'])}"
        )

        return splits

    def save_splits(
        self,
        splits: Dict[str, List[ParallelText]],
        output_dir: Union[str, Path],
        format: str = "jsonl",
    ) -> None:
        """Save train/dev/test splits to files.

        Args:
            splits: Dictionary of split names to parallel texts
            output_dir: Output directory for split files
            format: Output format ('jsonl' or 'json')

        Note:
            After saving splits, consider versioning with DVC:
            $ dvc add {output_dir}/train.jsonl
            $ dvc add {output_dir}/dev.jsonl
            $ dvc add {output_dir}/test.jsonl
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, texts in splits.items():
            output_file = output_dir / f"{split_name}.{format}"
            self.save_corpus(texts, output_file, format)

        self.logger.info(
            f"Consider versioning splits with DVC:\n"
            f"  dvc add {output_dir}/train.{format}\n"
            f"  dvc add {output_dir}/dev.{format}\n"
            f"  dvc add {output_dir}/test.{format}"
        )
