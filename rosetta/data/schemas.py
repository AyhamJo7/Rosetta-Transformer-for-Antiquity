"""Pydantic data models for the Rosetta Transformer project.

This module defines structured data schemas for documents, annotations, entities,
relations, and parallel texts used throughout the data processing pipeline.
All models use Pydantic for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class LanguageCode(str, Enum):
    """Supported language codes for ancient and modern languages."""

    # Ancient languages
    AKKADIAN = "akk"
    ANCIENT_EGYPTIAN = "egy"
    ANCIENT_GREEK = "grc"
    ARAMAIC = "arc"
    COPTIC = "cop"
    CUNEIFORM = "xcu"
    DEMOTIC = "egyd"
    HIERATIC = "egyh"
    HIEROGLYPHICS = "egyh"
    LATIN = "lat"
    OLD_PERSIAN = "peo"
    PHOENICIAN = "phn"
    SUMERIAN = "sux"
    UGARITIC = "uga"

    # Modern languages (for translations)
    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    ARABIC = "ar"
    ITALIAN = "it"
    SPANISH = "es"

    # Transliteration/romanization
    ROMANIZED = "rom"
    TRANSLITERATED = "trans"

    # Unknown or mixed
    UNKNOWN = "unk"
    MULTILINGUAL = "mul"


class DocumentSource(str, Enum):
    """Source types for documents."""

    OCR = "ocr"
    MANUAL_TRANSCRIPTION = "manual"
    DATABASE = "database"
    DIGITIZED_ARCHIVE = "archive"
    SCHOLARLY_EDITION = "scholarly"
    WEB_SCRAPE = "web"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class EntityLabel(str, Enum):
    """Named entity labels for ancient text annotation."""

    PERSON = "PERSON"
    DEITY = "DEITY"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    DATE = "DATE"
    TIME = "TIME"
    ARTIFACT = "ARTIFACT"
    RITUAL = "RITUAL"
    TITLE = "TITLE"
    NUMBER = "NUMBER"
    MEASURE = "MEASURE"
    EVENT = "EVENT"
    DYNASTY = "DYNASTY"
    TEMPLE = "TEMPLE"
    OFFERING = "OFFERING"
    OTHER = "OTHER"


class RelationLabel(str, Enum):
    """Relation types between entities in ancient texts."""

    RULER_OF = "RULER_OF"
    LOCATED_IN = "LOCATED_IN"
    BORN_IN = "BORN_IN"
    DIED_IN = "DIED_IN"
    WORSHIPPED_AT = "WORSHIPPED_AT"
    OFFERED_TO = "OFFERED_TO"
    PRECEDED_BY = "PRECEDED_BY"
    SUCCEEDED_BY = "SUCCEEDED_BY"
    PARENT_OF = "PARENT_OF"
    CHILD_OF = "CHILD_OF"
    SPOUSE_OF = "SPOUSE_OF"
    CONTEMPORARY_OF = "CONTEMPORARY_OF"
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    CREATED_BY = "CREATED_BY"
    DESTROYED_BY = "DESTROYED_BY"
    OTHER = "OTHER"


class Document(BaseModel):
    """Base document model for texts in the corpus.

    This model represents a single document with metadata, text content,
    and quality information. Documents can be from various sources including
    OCR, manual transcription, or databases.

    Attributes:
        id: Unique identifier for the document
        text: The actual text content of the document
        language: Language code for the document
        metadata: Additional metadata (e.g., period, provenance, genre)
        confidence_score: Quality/confidence score from 0.0 to 1.0
        source: Source type of the document
        created_at: Timestamp when the document was created
        updated_at: Timestamp of last update
    """

    id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., min_length=1, description="Document text content")
    language: LanguageCode = Field(..., description="Language of the text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence/quality score (0.0-1.0)",
    )
    source: DocumentSource = Field(
        default=DocumentSource.UNKNOWN, description="Document source type"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Ensure text is not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Text content cannot be empty or whitespace only")
        return v

    def word_count(self) -> int:
        """Count words in the document text.

        Returns:
            Number of whitespace-separated tokens
        """
        return len(self.text.split())

    def char_count(self) -> int:
        """Count characters in the document text.

        Returns:
            Number of characters including whitespace
        """
        return len(self.text)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "id": "doc_001",
                "text": "Ancient hieroglyphic text",
                "language": "egyh",
                "metadata": {"period": "New Kingdom", "source_tablet": "P.BM 10188"},
                "confidence_score": 0.95,
                "source": "ocr",
            }
        }


class Entity(BaseModel):
    """Named entity within a document.

    Represents a span of text that has been identified as a named entity
    with a specific label and confidence score.

    Attributes:
        text: The actual text span of the entity
        label: Entity type/category
        start: Character offset where entity starts
        end: Character offset where entity ends
        confidence: Confidence score for this entity
        normalized_form: Optional normalized or canonical form
    """

    text: str = Field(..., min_length=1, description="Entity text span")
    label: EntityLabel = Field(..., description="Entity type/category")
    start: int = Field(..., ge=0, description="Start character offset")
    end: int = Field(..., gt=0, description="End character offset")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )
    normalized_form: Optional[str] = Field(
        None, description="Normalized or canonical form of the entity"
    )

    @model_validator(mode="after")
    def validate_offsets(self) -> "Entity":
        """Ensure end offset is greater than start offset."""
        if self.end <= self.start:
            raise ValueError(
                f"End offset ({self.end}) must be greater than start offset ({self.start})"
            )
        return self

    def span(self) -> Tuple[int, int]:
        """Get the entity span as a tuple.

        Returns:
            Tuple of (start, end) offsets
        """
        return (self.start, self.end)

    def length(self) -> int:
        """Get the length of the entity span.

        Returns:
            Number of characters in the entity
        """
        return self.end - self.start

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "text": "Ramesses II",
                "label": "PERSON",
                "start": 0,
                "end": 11,
                "confidence": 0.98,
                "normalized_form": "Ramesses_II",
            }
        }


class Relation(BaseModel):
    """Relation between two entities in a document.

    Represents a typed relationship between a head entity and a tail entity,
    typically extracted through relation extraction models or rules.

    Attributes:
        head: The head entity in the relation
        tail: The tail entity in the relation
        label: Relation type
        confidence: Confidence score for this relation
        evidence_text: Optional text span supporting the relation
    """

    head: Entity = Field(..., description="Head entity in the relation")
    tail: Entity = Field(..., description="Tail entity in the relation")
    label: RelationLabel = Field(..., description="Relation type")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )
    evidence_text: Optional[str] = Field(
        None, description="Text span supporting this relation"
    )

    @model_validator(mode="after")
    def validate_different_entities(self) -> "Relation":
        """Ensure head and tail are different entities."""
        if (
            self.head.start == self.tail.start
            and self.head.end == self.tail.end
            and self.head.text == self.tail.text
        ):
            raise ValueError("Head and tail entities must be different")
        return self

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "head": {
                    "text": "Ramesses II",
                    "label": "PERSON",
                    "start": 0,
                    "end": 11,
                    "confidence": 0.98,
                },
                "tail": {
                    "text": "Egypt",
                    "label": "LOCATION",
                    "start": 20,
                    "end": 25,
                    "confidence": 0.99,
                },
                "label": "RULER_OF",
                "confidence": 0.95,
            }
        }


class AnnotatedDocument(Document):
    """Document with linguistic annotations.

    Extends the base Document model with additional fields for named entities,
    relations, and part-of-speech tags. Used for annotated training data.

    Attributes:
        entities: List of named entities in the document
        relations: List of relations between entities
        pos_tags: Optional POS tags as (token, tag) pairs
        parse_tree: Optional dependency parse tree
    """

    entities: List[Entity] = Field(
        default_factory=list, description="Named entities in the document"
    )
    relations: List[Relation] = Field(
        default_factory=list, description="Relations between entities"
    )
    pos_tags: List[Tuple[str, str]] = Field(
        default_factory=list, description="Part-of-speech tags as (token, tag) pairs"
    )
    parse_tree: Optional[Dict[str, Any]] = Field(
        None, description="Dependency parse tree structure"
    )

    @model_validator(mode="after")
    def validate_entity_offsets(self) -> "AnnotatedDocument":
        """Ensure all entity offsets are valid for the document text."""
        text_len = len(self.text)
        for entity in self.entities:
            if entity.end > text_len:
                raise ValueError(
                    f"Entity end offset {entity.end} exceeds document length {text_len}"
                )
            # Verify the entity text matches the span
            actual_text = self.text[entity.start : entity.end]
            if actual_text != entity.text:
                raise ValueError(
                    f"Entity text '{entity.text}' does not match "
                    f"document span '{actual_text}' at [{entity.start}:{entity.end}]"
                )
        return self

    def get_entities_by_label(self, label: EntityLabel) -> List[Entity]:
        """Get all entities with a specific label.

        Args:
            label: Entity label to filter by

        Returns:
            List of entities with the specified label
        """
        return [e for e in self.entities if e.label == label]

    def get_relations_by_label(self, label: RelationLabel) -> List[Relation]:
        """Get all relations with a specific label.

        Args:
            label: Relation label to filter by

        Returns:
            List of relations with the specified label
        """
        return [r for r in self.relations if r.label == label]

    def has_overlapping_entities(self) -> bool:
        """Check if any entities have overlapping spans.

        Returns:
            True if overlapping entities exist, False otherwise
        """
        sorted_entities = sorted(self.entities, key=lambda e: e.start)
        for i in range(len(sorted_entities) - 1):
            if sorted_entities[i].end > sorted_entities[i + 1].start:
                return True
        return False

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "id": "doc_001_annotated",
                "text": "Ramesses II ruled Egypt from Thebes",
                "language": "en",
                "entities": [
                    {"text": "Ramesses II", "label": "PERSON", "start": 0, "end": 11},
                    {"text": "Egypt", "label": "LOCATION", "start": 18, "end": 23},
                    {"text": "Thebes", "label": "LOCATION", "start": 29, "end": 35},
                ],
                "relations": [
                    {
                        "head": {
                            "text": "Ramesses II",
                            "label": "PERSON",
                            "start": 0,
                            "end": 11,
                        },
                        "tail": {
                            "text": "Egypt",
                            "label": "LOCATION",
                            "start": 18,
                            "end": 23,
                        },
                        "label": "RULER_OF",
                    }
                ],
                "source": "manual",
            }
        }


class ParallelText(BaseModel):
    """Parallel text pair for alignment and translation.

    Represents aligned texts in different languages or scripts, commonly used
    for training translation models or studying correspondences between ancient
    texts and their transliterations/translations.

    Attributes:
        source_text: Text in the source language
        target_text: Text in the target language
        alignment_score: Quality score for the alignment (0.0-1.0)
        language_pair: Tuple of (source_lang, target_lang)
        source_id: Optional ID linking to source document
        target_id: Optional ID linking to target document
        alignment_info: Optional detailed alignment information
        metadata: Additional metadata about the parallel text
    """

    source_text: str = Field(..., min_length=1, description="Source language text")
    target_text: str = Field(..., min_length=1, description="Target language text")
    alignment_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Alignment quality score (0.0-1.0)",
    )
    language_pair: Tuple[LanguageCode, LanguageCode] = Field(
        ..., description="(source_language, target_language) pair"
    )
    source_id: Optional[str] = Field(None, description="Source document ID")
    target_id: Optional[str] = Field(None, description="Target document ID")
    alignment_info: Optional[Dict[str, Any]] = Field(
        None, description="Detailed alignment information (e.g., word alignments)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("source_text", "target_text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Ensure texts are not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Text content cannot be empty or whitespace only")
        return v

    @model_validator(mode="after")
    def validate_language_pair(self) -> "ParallelText":
        """Ensure source and target languages are different."""
        source_lang, target_lang = self.language_pair
        if source_lang == target_lang:
            raise ValueError(
                f"Source and target languages must be different, got {source_lang}"
            )
        return self

    def length_ratio(self) -> float:
        """Calculate the length ratio between source and target texts.

        Returns:
            Ratio of target length to source length
        """
        if len(self.source_text) == 0:
            return 0.0
        return len(self.target_text) / len(self.source_text)

    def word_count_ratio(self) -> float:
        """Calculate the word count ratio between source and target texts.

        Returns:
            Ratio of target word count to source word count
        """
        source_words = len(self.source_text.split())
        target_words = len(self.target_text.split())
        if source_words == 0:
            return 0.0
        return target_words / source_words

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "source_text": "Ancient hieroglyphic text",
                "target_text": "imn-htp",
                "alignment_score": 0.95,
                "language_pair": ["egyh", "rom"],
                "source_id": "hier_001",
                "target_id": "rom_001",
                "metadata": {"alignment_method": "expert", "period": "New Kingdom"},
            }
        }


class CorpusStatistics(BaseModel):
    """Statistics about a text corpus.

    Provides summary statistics for analyzing corpus composition and quality.

    Attributes:
        total_documents: Total number of documents
        total_tokens: Total token count across all documents
        total_characters: Total character count
        language_distribution: Distribution of languages
        source_distribution: Distribution of document sources
        avg_confidence_score: Average confidence score
        avg_document_length: Average document length in tokens
    """

    total_documents: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    total_characters: int = Field(default=0, ge=0)
    language_distribution: Dict[str, int] = Field(default_factory=dict)
    source_distribution: Dict[str, int] = Field(default_factory=dict)
    avg_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_document_length: float = Field(default=0.0, ge=0.0)

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "total_documents": 1000,
                "total_tokens": 50000,
                "total_characters": 300000,
                "language_distribution": {"grc": 500, "lat": 300, "akk": 200},
                "source_distribution": {"ocr": 600, "manual": 300, "database": 100},
                "avg_confidence_score": 0.87,
                "avg_document_length": 50.0,
            }
        }
