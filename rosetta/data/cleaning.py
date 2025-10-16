"""Text cleaning and normalization utilities for ancient texts.

This module provides classes for cleaning, normalizing, and preprocessing
ancient text data, including handling damaged texts, unicode normalization,
deduplication, and sentence segmentation.
"""

import hashlib
import re
import unicodedata
from typing import Dict, List, Optional, Pattern, Set, Tuple, Union

import ftfy
from tqdm import tqdm

from rosetta.data.schemas import Document
from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


class UnicodeNormalizer:
    """Normalize unicode text for ancient scripts.

    Handles unicode normalization using various forms (NFC, NFD, NFKC, NFKD)
    and fixes common encoding issues with ftfy.
    """

    def __init__(
        self,
        normalization_form: str = "NFC",
        fix_encoding: bool = True,
        fix_entities: bool = True,
    ):
        """Initialize the unicode normalizer.

        Args:
            normalization_form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
            fix_encoding: Whether to fix encoding issues with ftfy
            fix_entities: Whether to fix HTML entities

        Raises:
            ValueError: If normalization_form is not valid
        """
        valid_forms = ["NFC", "NFD", "NFKC", "NFKD"]
        if normalization_form not in valid_forms:
            raise ValueError(
                f"Invalid normalization form: {normalization_form}. "
                f"Must be one of {valid_forms}"
            )

        self.normalization_form = normalization_form
        self.fix_encoding = fix_encoding
        self.fix_entities = fix_entities
        self.logger = get_logger(__name__)

    def normalize_text(self, text: str) -> str:
        """Normalize a single text string.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text
        """
        if not text:
            return text

        # Fix encoding issues first with ftfy
        if self.fix_encoding:
            text = ftfy.fix_text(text, fix_entities=self.fix_entities)

        # Apply unicode normalization
        from typing import cast

        text = unicodedata.normalize(cast(str, self.normalization_form), text)

        return text

    def normalize_document(self, document: Document) -> Document:
        """Normalize a document's text.

        Args:
            document: Document to normalize

        Returns:
            New Document with normalized text
        """
        normalized_text = self.normalize_text(document.text)

        # Create new document with normalized text
        return Document(
            id=document.id,
            text=normalized_text,
            language=document.language,
            metadata={
                **document.metadata,
                "normalized": True,
                "normalization_form": self.normalization_form,
            },
            confidence_score=document.confidence_score,
            source=document.source,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    def normalize_corpus(
        self, documents: List[Document], show_progress: bool = True
    ) -> List[Document]:
        """Normalize a corpus of documents.

        Args:
            documents: List of documents to normalize
            show_progress: Whether to show progress bar

        Returns:
            List of normalized documents
        """
        self.logger.info(
            f"Normalizing {len(documents)} documents with {self.normalization_form}"
        )

        doc_iter = tqdm(documents, desc="Normalizing") if show_progress else documents

        normalized = [self.normalize_document(doc) for doc in doc_iter]

        self.logger.info("Unicode normalization complete")
        return normalized


class DeduplicateTexts:
    """Deduplicate texts using exact and fuzzy matching.

    Removes duplicate documents from a corpus using exact hash matching
    and optional fuzzy similarity-based deduplication.
    """

    def __init__(
        self,
        method: str = "exact",
        similarity_threshold: float = 0.95,
        use_normalized: bool = True,
    ):
        """Initialize the deduplicator.

        Args:
            method: Deduplication method ('exact' or 'fuzzy')
            similarity_threshold: Similarity threshold for fuzzy matching (0.0-1.0)
            use_normalized: Whether to normalize text before comparison

        Raises:
            ValueError: If method is not 'exact' or 'fuzzy'
        """
        if method not in ["exact", "fuzzy"]:
            raise ValueError(f"Invalid method: {method}. Must be 'exact' or 'fuzzy'")

        self.method = method
        self.similarity_threshold = similarity_threshold
        self.use_normalized = use_normalized
        self.logger = get_logger(__name__)

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison.

        Args:
            text: Input text

        Returns:
            Normalized text for comparison
        """
        if not self.use_normalized:
            return text

        # Lowercase and remove extra whitespace
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def _compute_hash(self, text: str) -> str:
        """Compute hash of text for exact matching.

        Args:
            text: Input text

        Returns:
            SHA256 hash of the text
        """
        normalized = self._normalize_for_comparison(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity score (0.0-1.0)
        """
        # Tokenize into words
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def deduplicate_exact(
        self, documents: List[Document], show_progress: bool = True
    ) -> Tuple[List[Document], int]:
        """Remove exact duplicates based on text hash.

        Args:
            documents: List of documents to deduplicate
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (deduplicated documents, number of duplicates removed)
        """
        self.logger.info(f"Deduplicating {len(documents)} documents (exact matching)")

        seen_hashes: Set[str] = set()
        unique_docs: List[Document] = []
        duplicates_count = 0

        doc_iter = tqdm(documents, desc="Deduplicating") if show_progress else documents

        for doc in doc_iter:
            text_hash = self._compute_hash(doc.text)

            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_docs.append(doc)
            else:
                duplicates_count += 1
                self.logger.debug(f"Duplicate found: {doc.id}")

        self.logger.info(
            f"Removed {duplicates_count} exact duplicates, "
            f"{len(unique_docs)} unique documents remain"
        )

        return unique_docs, duplicates_count

    def deduplicate_fuzzy(
        self, documents: List[Document], show_progress: bool = True
    ) -> Tuple[List[Document], int]:
        """Remove fuzzy duplicates based on similarity threshold.

        Args:
            documents: List of documents to deduplicate
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (deduplicated documents, number of duplicates removed)

        Note:
            This is O(n^2) and can be slow for large corpora.
            For production use, consider using MinHash LSH or similar techniques.
        """
        self.logger.info(
            f"Deduplicating {len(documents)} documents "
            f"(fuzzy matching, threshold={self.similarity_threshold})"
        )

        unique_docs = []
        duplicates_count = 0

        doc_iter = tqdm(documents, desc="Deduplicating") if show_progress else documents

        for doc in doc_iter:
            normalized_text = self._normalize_for_comparison(doc.text)
            is_duplicate = False

            # Compare with all unique documents so far
            for unique_doc in unique_docs:
                unique_normalized = self._normalize_for_comparison(unique_doc.text)
                similarity = self._jaccard_similarity(
                    normalized_text, unique_normalized
                )

                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    duplicates_count += 1
                    self.logger.debug(
                        f"Fuzzy duplicate found: {doc.id} ~ {unique_doc.id} "
                        f"(similarity={similarity:.3f})"
                    )
                    break

            if not is_duplicate:
                unique_docs.append(doc)

        self.logger.info(
            f"Removed {duplicates_count} fuzzy duplicates, "
            f"{len(unique_docs)} unique documents remain"
        )

        return unique_docs, duplicates_count

    def deduplicate(
        self, documents: List[Document], show_progress: bool = True
    ) -> Tuple[List[Document], int]:
        """Deduplicate documents using configured method.

        Args:
            documents: List of documents to deduplicate
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (deduplicated documents, number of duplicates removed)
        """
        if self.method == "exact":
            return self.deduplicate_exact(documents, show_progress)
        else:
            return self.deduplicate_fuzzy(documents, show_progress)


class DamagedTextHandler:
    """Handle damaged or fragmentary ancient texts.

    Processes texts with missing sections, gaps, and damage markers
    commonly found in ancient manuscripts and inscriptions.
    """

    def __init__(
        self,
        gap_marker: str = "[...]",
        missing_marker: str = "[?]",
        damage_marker: str = "[ ]",
        preserve_markers: bool = True,
    ):
        """Initialize the damaged text handler.

        Args:
            gap_marker: Marker for text gaps
            missing_marker: Marker for uncertain/missing text
            damage_marker: Marker for damaged sections
            preserve_markers: Whether to preserve markers or remove them
        """
        self.gap_marker = gap_marker
        self.missing_marker = missing_marker
        self.damage_marker = damage_marker
        self.preserve_markers = preserve_markers
        self.logger = get_logger(__name__)

    def _standardize_markers(self, text: str) -> str:
        """Standardize various gap/damage notations to standard markers.

        Args:
            text: Input text with various damage notations

        Returns:
            Text with standardized markers
        """
        # Common gap notations: [...], &, ..., ---
        text = re.sub(r"\.{3,}|\u2026|+|-{3,}", self.gap_marker, text)

        # Common missing notations: [?], (?), [lost], [lacuna]
        text = re.sub(
            r"\[\?\]|\(\?\)|\[lost\]|\[lacuna\]",
            self.missing_marker,
            text,
            flags=re.IGNORECASE,
        )

        # Common damage notations: [ ], [corrupt], [damaged]
        text = re.sub(
            r"\[ \]|\[corrupt\]|\[damaged\]",
            self.damage_marker,
            text,
            flags=re.IGNORECASE,
        )

        return text

    def process_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Process a damaged text.

        Args:
            text: Input text with damage markers

        Returns:
            Tuple of (processed text, marker statistics)
        """
        # Standardize markers
        processed = self._standardize_markers(text)

        # Count markers
        stats = {
            "gaps": processed.count(self.gap_marker),
            "missing": processed.count(self.missing_marker),
            "damaged": processed.count(self.damage_marker),
        }

        # Remove markers if not preserving
        if not self.preserve_markers:
            processed = processed.replace(self.gap_marker, " ")
            processed = processed.replace(self.missing_marker, " ")
            processed = processed.replace(self.damage_marker, " ")

            # Clean up extra whitespace
            processed = re.sub(r"\s+", " ", processed).strip()

        return processed, stats

    def process_document(self, document: Document) -> Document:
        """Process a document with damaged text.

        Args:
            document: Document with potentially damaged text

        Returns:
            New Document with processed text and damage statistics
        """
        processed_text, stats = self.process_text(document.text)

        # Calculate damage ratio
        total_markers = sum(stats.values())
        word_count = len(document.text.split())
        damage_ratio = total_markers / word_count if word_count > 0 else 0.0

        return Document(
            id=document.id,
            text=processed_text,
            language=document.language,
            metadata={
                **document.metadata,
                "damage_stats": stats,
                "damage_ratio": damage_ratio,
                "markers_preserved": self.preserve_markers,
            },
            confidence_score=document.confidence_score * (1.0 - damage_ratio * 0.5),
            source=document.source,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    def process_corpus(
        self, documents: List[Document], show_progress: bool = True
    ) -> List[Document]:
        """Process a corpus of potentially damaged documents.

        Args:
            documents: List of documents to process
            show_progress: Whether to show progress bar

        Returns:
            List of processed documents
        """
        self.logger.info(f"Processing {len(documents)} documents for damage markers")

        doc_iter = tqdm(documents, desc="Processing") if show_progress else documents

        processed = [self.process_document(doc) for doc in doc_iter]

        self.logger.info("Damage marker processing complete")
        return processed


class CharsetNormalizer:
    """Normalize character sets for ancient scripts.

    Handles transliteration schemes and character variants in ancient scripts.
    """

    def __init__(self, transliteration_map: Optional[Dict[str, str]] = None):
        """Initialize the charset normalizer.

        Args:
            transliteration_map: Optional mapping for character normalization
        """
        self.transliteration_map = transliteration_map or self._default_maps()
        self.logger = get_logger(__name__)

    def _default_maps(self) -> Dict[str, str]:
        """Get default transliteration mappings.

        Returns:
            Dictionary of character mappings
        """
        return {
            # Greek variants (safely encoded)
            "ϐ": "β",  # Beta symbol to beta letter
            "ϑ": "θ",  # Theta symbol to theta letter
            "ϰ": "κ",  # Kappa symbol to kappa letter
            "ϱ": "ρ",  # Rho symbol to rho letter
            "ς": "σ",  # Final sigma to sigma
            # Common transliteration variations
            "ḥ": "h",  # h with dot below
            "ṭ": "t",  # t with dot below
            "ṣ": "s",  # s with dot below
            "š": "sh",  # s with caron
            "ś": "sh",  # s with acute
            # Combining diacritics normalization handled by unicode normalization
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text using transliteration map.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        for old_char, new_char in self.transliteration_map.items():
            text = text.replace(old_char, new_char)

        return text

    def normalize_document(self, document: Document) -> Document:
        """Normalize a document's character set.

        Args:
            document: Document to normalize

        Returns:
            New Document with normalized text
        """
        normalized_text = self.normalize_text(document.text)

        return Document(
            id=document.id,
            text=normalized_text,
            language=document.language,
            metadata={
                **document.metadata,
                "charset_normalized": True,
            },
            confidence_score=document.confidence_score,
            source=document.source,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    def normalize_corpus(
        self, documents: List[Document], show_progress: bool = True
    ) -> List[Document]:
        """Normalize a corpus of documents.

        Args:
            documents: List of documents to normalize
            show_progress: Whether to show progress bar

        Returns:
            List of normalized documents
        """
        self.logger.info(f"Normalizing character sets for {len(documents)} documents")

        doc_iter = tqdm(documents, desc="Normalizing") if show_progress else documents

        normalized = [self.normalize_document(doc) for doc in doc_iter]

        self.logger.info("Character set normalization complete")
        return normalized


class SentenceSegmenter:
    """Segment ancient texts into sentences.

    Provides sentence splitting tailored for ancient texts, which may not
    follow modern punctuation conventions.
    """

    def __init__(
        self,
        use_period: bool = True,
        use_semicolon: bool = True,
        use_question: bool = True,
        use_exclamation: bool = True,
        min_sentence_length: int = 5,
        custom_delimiters: Optional[List[str]] = None,
    ):
        """Initialize the sentence segmenter.

        Args:
            use_period: Split on periods
            use_semicolon: Split on semicolons
            use_question: Split on question marks
            use_exclamation: Split on exclamation marks
            min_sentence_length: Minimum sentence length in characters
            custom_delimiters: Additional custom delimiters
        """
        self.min_sentence_length = min_sentence_length
        self.logger = get_logger(__name__)

        # Build delimiter pattern
        delimiters = []
        if use_period:
            delimiters.append(r"\.")
        if use_semicolon:
            delimiters.append(r";")
        if use_question:
            delimiters.append(r"\?")
        if use_exclamation:
            delimiters.append(r"!")

        if custom_delimiters:
            delimiters.extend(re.escape(d) for d in custom_delimiters)

        if delimiters:
            self.delimiter_pattern: Optional[Pattern[str]] = re.compile(
                f"[{''.join(delimiters)}]"
            )
        else:
            self.delimiter_pattern = None

    def segment_text(self, text: str) -> List[str]:
        """Segment text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentence strings
        """
        if not text or not text.strip():
            return []

        if self.delimiter_pattern is None:
            # No delimiters, return whole text as one sentence
            return (
                [text.strip()] if len(text.strip()) >= self.min_sentence_length else []
            )

        # Split by delimiters
        sentences = self.delimiter_pattern.split(text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= self.min_sentence_length:
                cleaned_sentences.append(sent)

        return cleaned_sentences

    def segment_document(self, document: Document) -> Tuple[List[str], Dict[str, int]]:
        """Segment a document into sentences.

        Args:
            document: Document to segment

        Returns:
            Tuple of (list of sentences, statistics)
        """
        sentences = self.segment_text(document.text)

        stats: Dict[str, Union[int, float]] = {
            "num_sentences": len(sentences),
            "avg_sentence_length": (
                sum(len(s) for s in sentences) / len(sentences) if sentences else 0
            ),
            "min_sentence_length": min(len(s) for s in sentences) if sentences else 0,
            "max_sentence_length": max(len(s) for s in sentences) if sentences else 0,
        }

        return sentences, stats

    def segment_corpus(
        self, documents: List[Document], show_progress: bool = True
    ) -> Dict[str, List[str]]:
        """Segment a corpus into sentences.

        Args:
            documents: List of documents to segment
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping document IDs to sentence lists
        """
        self.logger.info(f"Segmenting {len(documents)} documents into sentences")

        doc_iter = tqdm(documents, desc="Segmenting") if show_progress else documents

        corpus_sentences = {}
        total_sentences = 0

        for doc in doc_iter:
            sentences, stats = self.segment_document(doc)
            corpus_sentences[doc.id] = sentences
            total_sentences += len(sentences)

        self.logger.info(
            f"Segmentation complete: {total_sentences} sentences from "
            f"{len(documents)} documents"
        )

        return corpus_sentences
