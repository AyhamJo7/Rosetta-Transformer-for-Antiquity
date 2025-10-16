"""Corpus curation and loading utilities for ancient texts.

This module provides classes for loading, processing, and building corpora
from various data sources including text files, JSON, XML, and OCR outputs.
Supports DVC for data versioning.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import jsonlines
from tqdm import tqdm

from rosetta.data.schemas import (
    CorpusStatistics,
    Document,
    DocumentSource,
    LanguageCode,
)
from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


class CorpusLoader:
    """Load documents from various file formats.

    Supports loading text corpora from:
    - Plain text files (.txt)
    - JSON files (.json)
    - JSON Lines files (.jsonl)
    - XML files (.xml)

    Each loader method yields Document objects for efficient memory usage
    when dealing with large corpora.

    Note:
        For DVC integration, ensure corpus files are tracked with DVC:
        $ dvc add data/corpus.jsonl
        $ git add data/corpus.jsonl.dvc data/.gitignore
    """

    def __init__(self, default_language: LanguageCode = LanguageCode.UNKNOWN):
        """Initialize the corpus loader.

        Args:
            default_language: Default language code when not specified in data
        """
        self.default_language = default_language
        self.logger = get_logger(__name__)

    def load_txt(
        self,
        file_path: Union[str, Path],
        document_separator: str = "\n\n",
        language: Optional[LanguageCode] = None,
        source: DocumentSource = DocumentSource.UNKNOWN,
    ) -> Iterator[Document]:
        """Load documents from a plain text file.

        Documents are separated by a delimiter (default: double newline).
        Each document gets a sequential ID.

        Args:
            file_path: Path to the text file
            document_separator: String that separates documents
            language: Language code for the documents
            source: Source type for the documents

        Yields:
            Document objects parsed from the text file

        Raises:
            FileNotFoundError: If file_path does not exist
            ValueError: If file is empty or cannot be decoded
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.logger.info(f"Loading text documents from {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode {file_path} as UTF-8: {e}")

        if not content.strip():
            raise ValueError(f"File is empty: {file_path}")

        # Split into documents
        texts = content.split(document_separator)
        lang = language or self.default_language

        for idx, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue

            yield Document(
                id=f"{file_path.stem}_{idx:06d}",
                text=text,
                language=lang,
                source=source,
                metadata={"source_file": str(file_path), "index": idx},
            )

        self.logger.info(f"Loaded {idx + 1} documents from {file_path}")

    def load_json(
        self,
        file_path: Union[str, Path],
        text_field: str = "text",
        id_field: str = "id",
        language_field: str = "language",
        metadata_fields: Optional[List[str]] = None,
    ) -> Iterator[Document]:
        """Load documents from a JSON file.

        Expects a JSON array of objects, each representing a document.

        Args:
            file_path: Path to the JSON file
            text_field: Name of the field containing document text
            id_field: Name of the field containing document ID
            language_field: Name of the field containing language code
            metadata_fields: Additional fields to include in metadata

        Yields:
            Document objects parsed from the JSON file

        Raises:
            FileNotFoundError: If file_path does not exist
            ValueError: If JSON is invalid or missing required fields
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.logger.info(f"Loading JSON documents from {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")

        if not isinstance(data, list):
            raise ValueError(f"JSON root must be an array, got {type(data)}")

        metadata_fields = metadata_fields or []

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                self.logger.warning(f"Skipping non-dict item at index {idx}")
                continue

            # Extract required fields
            text = item.get(text_field)
            if not text:
                self.logger.warning(f"Skipping item {idx}: missing '{text_field}'")
                continue

            doc_id = item.get(id_field, f"json_doc_{idx:06d}")
            language = item.get(language_field, self.default_language)

            # Try to convert language string to enum
            if isinstance(language, str):
                try:
                    language = LanguageCode(language)
                except ValueError:
                    self.logger.warning(
                        f"Unknown language code '{language}', using default"
                    )
                    language = self.default_language

            # Extract metadata
            metadata = {"source_file": str(file_path), "index": idx}
            for field in metadata_fields:
                if field in item:
                    metadata[field] = item[field]

            # Get source and confidence if available
            source = item.get("source", DocumentSource.UNKNOWN)
            if isinstance(source, str):
                try:
                    source = DocumentSource(source)
                except ValueError:
                    source = DocumentSource.UNKNOWN

            confidence = item.get("confidence_score", 1.0)

            yield Document(
                id=doc_id,
                text=text,
                language=language,
                metadata=metadata,
                confidence_score=confidence,
                source=source,
            )

        self.logger.info(f"Loaded {idx + 1} documents from {file_path}")

    def load_jsonl(
        self,
        file_path: Union[str, Path],
        text_field: str = "text",
        id_field: str = "id",
        language_field: str = "language",
        metadata_fields: Optional[List[str]] = None,
    ) -> Iterator[Document]:
        """Load documents from a JSON Lines file.

        Each line is a separate JSON object representing a document.
        More memory-efficient than load_json for large corpora.

        Args:
            file_path: Path to the JSONL file
            text_field: Name of the field containing document text
            id_field: Name of the field containing document ID
            language_field: Name of the field containing language code
            metadata_fields: Additional fields to include in metadata

        Yields:
            Document objects parsed from the JSONL file

        Raises:
            FileNotFoundError: If file_path does not exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.logger.info(f"Loading JSONL documents from {file_path}")

        metadata_fields = metadata_fields or []
        count = 0

        with jsonlines.open(file_path) as reader:
            for idx, item in enumerate(reader):
                if not isinstance(item, dict):
                    self.logger.warning(f"Skipping non-dict item at line {idx + 1}")
                    continue

                # Extract required fields
                text = item.get(text_field)
                if not text:
                    self.logger.warning(
                        f"Skipping line {idx + 1}: missing '{text_field}'"
                    )
                    continue

                doc_id = item.get(id_field, f"jsonl_doc_{idx:06d}")
                language = item.get(language_field, self.default_language)

                # Try to convert language string to enum
                if isinstance(language, str):
                    try:
                        language = LanguageCode(language)
                    except ValueError:
                        language = self.default_language

                # Extract metadata
                metadata = {"source_file": str(file_path), "line": idx + 1}
                for field in metadata_fields:
                    if field in item:
                        metadata[field] = item[field]

                # Get source and confidence if available
                source = item.get("source", DocumentSource.UNKNOWN)
                if isinstance(source, str):
                    try:
                        source = DocumentSource(source)
                    except ValueError:
                        source = DocumentSource.UNKNOWN

                confidence = item.get("confidence_score", 1.0)

                yield Document(
                    id=doc_id,
                    text=text,
                    language=language,
                    metadata=metadata,
                    confidence_score=confidence,
                    source=source,
                )
                count += 1

        self.logger.info(f"Loaded {count} documents from {file_path}")

    def load_xml(
        self,
        file_path: Union[str, Path],
        document_tag: str = "document",
        text_tag: str = "text",
        id_attr: str = "id",
        language_attr: str = "language",
    ) -> Iterator[Document]:
        """Load documents from an XML file.

        Expects XML structure like:
        <corpus>
            <document id="doc1" language="grc">
                <text>Ancient Greek text here</text>
            </document>
        </corpus>

        Args:
            file_path: Path to the XML file
            document_tag: XML tag name for document elements
            text_tag: XML tag name for text content
            id_attr: Attribute name for document ID
            language_attr: Attribute name for language code

        Yields:
            Document objects parsed from the XML file

        Raises:
            FileNotFoundError: If file_path does not exist
            ValueError: If XML is malformed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.logger.info(f"Loading XML documents from {file_path}")

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML in {file_path}: {e}")

        count = 0
        for idx, doc_elem in enumerate(root.findall(f".//{document_tag}")):
            # Get document ID
            doc_id = doc_elem.get(id_attr, f"xml_doc_{idx:06d}")

            # Get language
            language_str = doc_elem.get(language_attr, self.default_language.value)
            try:
                language = LanguageCode(language_str)
            except ValueError:
                language = self.default_language

            # Get text content
            text_elem = doc_elem.find(text_tag)
            if text_elem is None or not text_elem.text:
                self.logger.warning(f"Skipping document {doc_id}: no text found")
                continue

            text = text_elem.text.strip()

            # Extract metadata from other attributes
            metadata = {
                "source_file": str(file_path),
                "index": idx,
            }
            for attr_name, attr_value in doc_elem.attrib.items():
                if attr_name not in [id_attr, language_attr]:
                    metadata[attr_name] = attr_value

            yield Document(
                id=doc_id,
                text=text,
                language=language,
                metadata=metadata,
                source=DocumentSource.DATABASE,
            )
            count += 1

        self.logger.info(f"Loaded {count} documents from {file_path}")


class OCRProcessor:
    """Process OCR data with confidence filtering.

    Handles OCR output formats and filters low-confidence texts.
    Useful for ancient manuscripts and inscriptions.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        min_word_confidence: float = 0.3,
    ):
        """Initialize the OCR processor.

        Args:
            min_confidence: Minimum document-level confidence (0.0-1.0)
            min_word_confidence: Minimum word-level confidence (0.0-1.0)
        """
        self.min_confidence = min_confidence
        self.min_word_confidence = min_word_confidence
        self.logger = get_logger(__name__)

    def process_ocr_document(
        self,
        text: str,
        confidence_scores: Optional[List[float]] = None,
        doc_id: Optional[str] = None,
        language: LanguageCode = LanguageCode.UNKNOWN,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Document]:
        """Process a single OCR document with confidence filtering.

        Args:
            text: OCR text output
            confidence_scores: Per-word confidence scores (0.0-1.0)
            doc_id: Document identifier
            language: Language code
            metadata: Additional metadata

        Returns:
            Document if confidence is sufficient, None otherwise
        """
        if not text or not text.strip():
            return None

        # Calculate document confidence
        if confidence_scores:
            doc_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            doc_confidence = 1.0

        # Filter by document-level confidence
        if doc_confidence < self.min_confidence:
            self.logger.debug(
                f"Skipping document {doc_id}: confidence {doc_confidence:.2f} "
                f"below threshold {self.min_confidence}"
            )
            return None

        # Filter low-confidence words if scores are available
        if confidence_scores:
            words = text.split()
            if len(words) != len(confidence_scores):
                self.logger.warning(
                    f"Word count ({len(words)}) != confidence count "
                    f"({len(confidence_scores)}) for {doc_id}"
                )
            else:
                # Filter out low-confidence words
                filtered_words = [
                    word
                    for word, conf in zip(words, confidence_scores)
                    if conf >= self.min_word_confidence
                ]
                text = " ".join(filtered_words)

        if not text.strip():
            return None

        metadata = metadata or {}
        metadata["ocr_confidence"] = doc_confidence

        return Document(
            id=doc_id or f"ocr_doc_{hash(text) % 1000000:06d}",
            text=text,
            language=language,
            metadata=metadata,
            confidence_score=doc_confidence,
            source=DocumentSource.OCR,
        )

    def process_hocr(self, hocr_file: Union[str, Path]) -> Iterator[Document]:
        """Process hOCR (HTML-based OCR) format.

        Args:
            hocr_file: Path to hOCR file

        Yields:
            Document objects extracted from hOCR

        Note:
            This is a simplified implementation. For production use,
            consider using specialized libraries like hocr-tools.
        """
        # Placeholder for hOCR processing
        # In production, parse hOCR XML and extract confidence scores
        self.logger.warning("hOCR processing is not fully implemented")
        yield from []


class MetadataNormalizer:
    """Normalize metadata across different data sources.

    Standardizes metadata fields from various sources to a common schema.
    Useful when combining corpora from multiple origins.
    """

    def __init__(self, field_mappings: Optional[Dict[str, str]] = None):
        """Initialize the metadata normalizer.

        Args:
            field_mappings: Dictionary mapping source fields to standard fields
                           e.g., {"author": "creator", "date": "creation_date"}
        """
        self.field_mappings = field_mappings or {}
        self.logger = get_logger(__name__)

    def normalize(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize metadata fields.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Normalized metadata dictionary
        """
        normalized = {}

        for key, value in metadata.items():
            # Apply field mapping if exists
            normalized_key = self.field_mappings.get(key, key)

            # Normalize common fields
            if normalized_key in ["date", "creation_date", "year"]:
                value = self._normalize_date(value)
            elif normalized_key in ["period", "era"]:
                value = self._normalize_period(value)
            elif normalized_key in ["language", "lang"]:
                value = self._normalize_language(value)

            normalized[normalized_key] = value

        return normalized

    def _normalize_date(self, date_value: Any) -> str:
        """Normalize date formats.

        Args:
            date_value: Date in various formats

        Returns:
            Standardized date string
        """
        # Simple string conversion for now
        # In production, handle various date formats
        return str(date_value)

    def _normalize_period(self, period_value: Any) -> str:
        """Normalize period/era names.

        Args:
            period_value: Period name

        Returns:
            Standardized period name
        """
        # Normalize period names (e.g., "NK" -> "New Kingdom")
        period_map = {
            "NK": "New Kingdom",
            "MK": "Middle Kingdom",
            "OK": "Old Kingdom",
            "LP": "Late Period",
            "PT": "Ptolemaic",
        }
        period_str = str(period_value).upper()
        return str(period_map.get(period_str, period_value))

    def _normalize_language(self, language_value: Any) -> str:
        """Normalize language codes.

        Args:
            language_value: Language code or name

        Returns:
            Standardized language code
        """
        # Map common variations to standard codes
        lang_map = {
            "ancient greek": "grc",
            "greek": "grc",
            "latin": "lat",
            "egyptian": "egy",
            "hieroglyphic": "egyh",
            "akkadian": "akk",
        }
        lang_str = str(language_value).lower()
        return str(lang_map.get(lang_str, language_value))


class CorpusBuilder:
    """Build and manage text corpora from multiple sources.

    Combines CorpusLoader, OCRProcessor, and MetadataNormalizer
    to create unified, quality-filtered corpora.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        metadata_normalizer: Optional[MetadataNormalizer] = None,
    ):
        """Initialize the corpus builder.

        Args:
            min_confidence: Minimum confidence threshold for OCR documents
            metadata_normalizer: Optional normalizer for metadata
        """
        self.loader = CorpusLoader()
        self.ocr_processor = OCRProcessor(min_confidence=min_confidence)
        self.metadata_normalizer = metadata_normalizer or MetadataNormalizer()
        self.logger = get_logger(__name__)

    def build_from_files(
        self,
        file_paths: List[Union[str, Path]],
        output_file: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
    ) -> List[Document]:
        """Build corpus from multiple files.

        Args:
            file_paths: List of paths to corpus files
            output_file: Optional path to save the combined corpus (JSONL)
            show_progress: Whether to show progress bar

        Returns:
            List of Document objects

        Note:
            After building corpus, consider versioning with DVC:
            $ dvc add {output_file}
        """
        documents = []

        file_list = (
            tqdm(file_paths, desc="Loading files") if show_progress else file_paths
        )

        for file_path in file_list:
            file_path = Path(file_path)

            try:
                if file_path.suffix == ".txt":
                    docs = self.loader.load_txt(file_path)
                elif file_path.suffix == ".json":
                    docs = self.loader.load_json(file_path)
                elif file_path.suffix == ".jsonl":
                    docs = self.loader.load_jsonl(file_path)
                elif file_path.suffix == ".xml":
                    docs = self.loader.load_xml(file_path)
                else:
                    self.logger.warning(f"Unsupported file format: {file_path}")
                    continue

                for doc in docs:
                    # Normalize metadata
                    doc.metadata = self.metadata_normalizer.normalize(doc.metadata)
                    documents.append(doc)

            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                continue

        self.logger.info(f"Built corpus with {len(documents)} documents")

        # Save to output file if specified
        if output_file:
            self.save_corpus(documents, output_file)
            self.logger.info(f"Saved corpus to {output_file}")
            self.logger.info(f"Consider versioning with DVC: dvc add {output_file}")

        return documents

    def save_corpus(
        self,
        documents: List[Document],
        output_file: Union[str, Path],
        format: str = "jsonl",
    ) -> None:
        """Save corpus to file.

        Args:
            documents: List of documents to save
            output_file: Path to output file
            format: Output format ('jsonl' or 'json')

        Raises:
            ValueError: If format is not supported
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with jsonlines.open(output_file, mode="w") as writer:
                for doc in documents:
                    writer.write(doc.model_dump())
        elif format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    [doc.model_dump() for doc in documents],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_statistics(self, documents: List[Document]) -> CorpusStatistics:
        """Compute corpus statistics.

        Args:
            documents: List of documents

        Returns:
            CorpusStatistics object with summary information
        """
        if not documents:
            return CorpusStatistics()

        total_tokens = sum(doc.word_count() for doc in documents)
        total_chars = sum(doc.char_count() for doc in documents)

        # Language distribution
        lang_dist: Dict[str, int] = {}
        for doc in documents:
            lang_key = (
                doc.language.value
                if hasattr(doc.language, "value")
                else str(doc.language)
            )
            lang_dist[lang_key] = lang_dist.get(lang_key, 0) + 1

        # Source distribution
        source_dist: Dict[str, int] = {}
        for doc in documents:
            source_key = (
                doc.source.value if hasattr(doc.source, "value") else str(doc.source)
            )
            source_dist[source_key] = source_dist.get(source_key, 0) + 1

        # Average confidence
        avg_confidence = sum(doc.confidence_score for doc in documents) / len(documents)

        # Average document length
        avg_length = total_tokens / len(documents) if documents else 0.0

        return CorpusStatistics(
            total_documents=len(documents),
            total_tokens=total_tokens,
            total_characters=total_chars,
            language_distribution=lang_dist,
            source_distribution=source_dist,
            avg_confidence_score=avg_confidence,
            avg_document_length=avg_length,
        )
