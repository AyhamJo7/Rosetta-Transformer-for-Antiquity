"""Data processing modules for the Rosetta Transformer project.

This package provides comprehensive data processing utilities for ancient texts,
including:

- **schemas**: Pydantic data models for documents, entities, relations, and parallel texts
- **corpus**: Corpus loading and curation from various formats
- **cleaning**: Text cleaning, normalization, and preprocessing
- **annotation**: NER and relation extraction pipelines
- **alignment**: Parallel text alignment and quality scoring

Example Usage:
    >>> from rosetta.data import Document, CorpusLoader, UnicodeNormalizer
    >>>
    >>> # Load a corpus
    >>> loader = CorpusLoader()
    >>> documents = list(loader.load_jsonl("corpus.jsonl"))
    >>>
    >>> # Clean and normalize
    >>> normalizer = UnicodeNormalizer(normalization_form="NFC")
    >>> clean_docs = normalizer.normalize_corpus(documents)
    >>>
    >>> # Annotate with NER
    >>> from rosetta.data import NERAnnotator
    >>> annotator = NERAnnotator(method="rule-based")
    >>> annotated = annotator.annotate_corpus(clean_docs)
"""

# Schema exports
# Alignment exports
from rosetta.data.alignment import (
    AlignmentScorer,
    ParallelCorpusBuilder,
    TextAligner,
)

# Annotation exports
from rosetta.data.annotation import (
    AnnotationValidator,
    GoldSetBuilder,
    NERAnnotator,
    REAnnotator,
)

# Cleaning exports
from rosetta.data.cleaning import (
    CharsetNormalizer,
    DamagedTextHandler,
    DeduplicateTexts,
    SentenceSegmenter,
    UnicodeNormalizer,
)

# Corpus loading exports
from rosetta.data.corpus import (
    CorpusBuilder,
    CorpusLoader,
    MetadataNormalizer,
    OCRProcessor,
)
from rosetta.data.schemas import (
    AnnotatedDocument,
    CorpusStatistics,
    Document,
    DocumentSource,
    Entity,
    EntityLabel,
    LanguageCode,
    ParallelText,
    Relation,
    RelationLabel,
)

__all__ = [
    # Schemas
    "Document",
    "AnnotatedDocument",
    "Entity",
    "Relation",
    "ParallelText",
    "CorpusStatistics",
    "LanguageCode",
    "DocumentSource",
    "EntityLabel",
    "RelationLabel",
    # Corpus
    "CorpusLoader",
    "OCRProcessor",
    "MetadataNormalizer",
    "CorpusBuilder",
    # Cleaning
    "UnicodeNormalizer",
    "DeduplicateTexts",
    "DamagedTextHandler",
    "CharsetNormalizer",
    "SentenceSegmenter",
    # Annotation
    "NERAnnotator",
    "REAnnotator",
    "AnnotationValidator",
    "GoldSetBuilder",
    # Alignment
    "TextAligner",
    "AlignmentScorer",
    "ParallelCorpusBuilder",
]
