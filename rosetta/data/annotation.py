"""Annotation pipelines for ancient text NER and relation extraction.

This module provides classes for annotating ancient texts with named entities,
relations, and other linguistic information. Supports both automated
annotation with spaCy and rule-based methods for ancient languages.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from rosetta.data.schemas import (
    AnnotatedDocument,
    Document,
    Entity,
    EntityLabel,
    Relation,
    RelationLabel,
)
from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


class NERAnnotator:
    """Named Entity Recognition annotator for ancient texts.

    Supports both spaCy-based NER for modern languages and rule-based
    NER for ancient languages where trained models may not exist.
    """

    def __init__(
        self,
        method: str = "spacy",
        spacy_model: Optional[str] = None,
        custom_rules: Optional[Dict[EntityLabel, List[str]]] = None,
        min_confidence: float = 0.5,
    ):
        """Initialize the NER annotator.

        Args:
            method: Annotation method ('spacy' or 'rule-based')
            spacy_model: spaCy model name (e.g., 'en_core_web_sm')
            custom_rules: Dictionary mapping entity labels to keyword lists
            min_confidence: Minimum confidence threshold for entities

        Raises:
            ValueError: If method is not supported
        """
        if method not in ["spacy", "rule-based"]:
            raise ValueError(f"Unsupported method: {method}")

        self.method = method
        self.min_confidence = min_confidence
        self.custom_rules = custom_rules or {}
        self.logger = get_logger(__name__)

        # Initialize spaCy model if needed
        self.nlp = None
        if method == "spacy":
            if not spacy_model:
                raise ValueError("spacy_model must be provided for spacy method")

            try:
                import spacy

                self.nlp = spacy.load(spacy_model)
                self.logger.info(f"Loaded spaCy model: {spacy_model}")
            except ImportError:
                raise ImportError(
                    "spaCy is required for spacy method. Install with: pip install spacy"
                )
            except OSError:
                raise OSError(
                    f"spaCy model '{spacy_model}' not found. "
                    f"Download with: python -m spacy download {spacy_model}"
                )

    def _map_spacy_label(self, spacy_label: str) -> EntityLabel:
        """Map spaCy entity labels to our schema.

        Args:
            spacy_label: spaCy entity label

        Returns:
            Corresponding EntityLabel
        """
        label_map = {
            "PERSON": EntityLabel.PERSON,
            "PER": EntityLabel.PERSON,
            "GPE": EntityLabel.LOCATION,
            "LOC": EntityLabel.LOCATION,
            "ORG": EntityLabel.ORGANIZATION,
            "DATE": EntityLabel.DATE,
            "TIME": EntityLabel.TIME,
            "NORP": EntityLabel.ORGANIZATION,
            "FAC": EntityLabel.LOCATION,
            "PRODUCT": EntityLabel.ARTIFACT,
            "EVENT": EntityLabel.EVENT,
            "WORK_OF_ART": EntityLabel.ARTIFACT,
            "CARDINAL": EntityLabel.NUMBER,
            "ORDINAL": EntityLabel.NUMBER,
            "QUANTITY": EntityLabel.MEASURE,
        }

        return label_map.get(spacy_label, EntityLabel.OTHER)

    def annotate_with_spacy(self, document: Document) -> AnnotatedDocument:
        """Annotate document using spaCy.

        Args:
            document: Document to annotate

        Returns:
            AnnotatedDocument with entities extracted by spaCy
        """
        if self.nlp is None:
            raise ValueError("spaCy model not initialized")

        doc = self.nlp(document.text)
        entities = []

        for ent in doc.ents:
            # Map spaCy label to our schema
            label = self._map_spacy_label(ent.label_)

            # Create entity
            entity = Entity(  # type: ignore[call-arg]
                text=ent.text,
                label=label,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,  # spaCy doesn't provide confidence scores by default
            )

            entities.append(entity)

        # Extract POS tags
        pos_tags = [(token.text, token.pos_) for token in doc]

        return AnnotatedDocument(  # type: ignore[call-arg]
            id=document.id,
            text=document.text,
            language=document.language,
            metadata={**document.metadata, "annotation_method": "spacy"},
            confidence_score=document.confidence_score,
            source=document.source,
            entities=entities,
            pos_tags=pos_tags,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    def annotate_with_rules(self, document: Document) -> AnnotatedDocument:
        """Annotate document using rule-based matching.

        Args:
            document: Document to annotate

        Returns:
            AnnotatedDocument with entities extracted by rules
        """
        entities = []
        text_lower = document.text.lower()

        # Apply custom rules
        for label, keywords in self.custom_rules.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()

                # Find all occurrences of keyword
                start = 0
                while True:
                    idx = text_lower.find(keyword_lower, start)
                    if idx == -1:
                        break

                    # Get actual text (preserving case)
                    entity_text = document.text[idx : idx + len(keyword)]

                    entity = Entity(  # type: ignore[call-arg]
                        text=entity_text,
                        label=label,
                        start=idx,
                        end=idx + len(keyword),
                        confidence=0.8,  # Rule-based confidence
                    )

                    entities.append(entity)
                    start = idx + len(keyword)

        # Remove overlapping entities (keep higher confidence)
        entities = self._remove_overlapping_entities(entities)

        return AnnotatedDocument(  # type: ignore[call-arg]
            id=document.id,
            text=document.text,
            language=document.language,
            metadata={**document.metadata, "annotation_method": "rule-based"},
            confidence_score=document.confidence_score,
            source=document.source,
            entities=entities,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    def _remove_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping higher confidence ones.

        Args:
            entities: List of entities that may overlap

        Returns:
            List of non-overlapping entities
        """
        if not entities:
            return []

        # Sort by start position, then by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: (e.start, -e.confidence))

        non_overlapping = []
        last_end = -1

        for entity in sorted_entities:
            if entity.start >= last_end:
                non_overlapping.append(entity)
                last_end = entity.end

        return non_overlapping

    def annotate_document(self, document: Document) -> AnnotatedDocument:
        """Annotate a document using the configured method.

        Args:
            document: Document to annotate

        Returns:
            AnnotatedDocument with extracted entities
        """
        if self.method == "spacy":
            return self.annotate_with_spacy(document)
        else:
            return self.annotate_with_rules(document)

    def annotate_corpus(
        self, documents: List[Document], show_progress: bool = True
    ) -> List[AnnotatedDocument]:
        """Annotate a corpus of documents.

        Args:
            documents: List of documents to annotate
            show_progress: Whether to show progress bar

        Returns:
            List of annotated documents
        """
        self.logger.info(
            f"Annotating {len(documents)} documents with {self.method} method"
        )

        doc_iter = tqdm(documents, desc="Annotating") if show_progress else documents

        annotated = [self.annotate_document(doc) for doc in doc_iter]

        total_entities = sum(len(doc.entities) for doc in annotated)
        self.logger.info(
            f"Annotation complete: {total_entities} entities extracted "
            f"from {len(documents)} documents"
        )

        return annotated


class REAnnotator:
    """Relation Extraction annotator for ancient texts.

    Extracts relations between entities using rule-based patterns
    or simple heuristics based on entity proximity and types.
    """

    def __init__(
        self,
        max_entity_distance: int = 50,
        relation_rules: Optional[
            Dict[Tuple[EntityLabel, EntityLabel], RelationLabel]
        ] = None,
        min_confidence: float = 0.5,
    ):
        """Initialize the relation extractor.

        Args:
            max_entity_distance: Maximum character distance between entities
            relation_rules: Dictionary mapping entity type pairs to relation types
            min_confidence: Minimum confidence threshold for relations
        """
        self.max_entity_distance = max_entity_distance
        self.min_confidence = min_confidence
        self.logger = get_logger(__name__)

        # Default relation rules based on entity types
        self.relation_rules = relation_rules or {
            (EntityLabel.PERSON, EntityLabel.LOCATION): RelationLabel.LOCATED_IN,
            (EntityLabel.PERSON, EntityLabel.ORGANIZATION): RelationLabel.PART_OF,
            (EntityLabel.DEITY, EntityLabel.TEMPLE): RelationLabel.WORSHIPPED_AT,
            (EntityLabel.PERSON, EntityLabel.DEITY): RelationLabel.WORSHIPPED_AT,
            (EntityLabel.OFFERING, EntityLabel.DEITY): RelationLabel.OFFERED_TO,
            (EntityLabel.ARTIFACT, EntityLabel.LOCATION): RelationLabel.LOCATED_IN,
            (EntityLabel.EVENT, EntityLabel.LOCATION): RelationLabel.LOCATED_IN,
            (EntityLabel.EVENT, EntityLabel.DATE): RelationLabel.PART_OF,
        }

    def extract_relations(self, annotated_doc: AnnotatedDocument) -> List[Relation]:
        """Extract relations from an annotated document.

        Args:
            annotated_doc: Document with entity annotations

        Returns:
            List of extracted relations
        """
        relations = []
        entities = annotated_doc.entities

        # Check all entity pairs
        for i, head in enumerate(entities):
            for tail in entities[i + 1 :]:
                # Check distance between entities
                distance = min(
                    abs(tail.start - head.end),
                    abs(head.start - tail.end),
                )

                if distance > self.max_entity_distance:
                    continue

                # Check if there's a rule for this entity pair
                entity_pair = (head.label, tail.label)
                reverse_pair = (tail.label, head.label)

                relation_label = None
                is_reversed = False

                if entity_pair in self.relation_rules:
                    relation_label = self.relation_rules[entity_pair]
                elif reverse_pair in self.relation_rules:
                    relation_label = self.relation_rules[reverse_pair]
                    is_reversed = True

                if relation_label:
                    # Calculate confidence based on distance
                    confidence = max(
                        0.5, 1.0 - (distance / self.max_entity_distance) * 0.5
                    )

                    if confidence >= self.min_confidence:
                        # Swap head and tail if reversed
                        if is_reversed:
                            head, tail = tail, head

                        relation = Relation(
                            head=head,
                            tail=tail,
                            label=relation_label,
                            confidence=confidence,
                        )

                        relations.append(relation)

        return relations

    def annotate_document(self, annotated_doc: AnnotatedDocument) -> AnnotatedDocument:
        """Add relation annotations to a document.

        Args:
            annotated_doc: Document with entity annotations

        Returns:
            AnnotatedDocument with both entities and relations
        """
        relations = self.extract_relations(annotated_doc)

        return AnnotatedDocument(
            id=annotated_doc.id,
            text=annotated_doc.text,
            language=annotated_doc.language,
            metadata={
                **annotated_doc.metadata,
                "has_relation_extraction": True,
            },
            confidence_score=annotated_doc.confidence_score,
            source=annotated_doc.source,
            entities=annotated_doc.entities,
            relations=relations,
            pos_tags=annotated_doc.pos_tags,
            parse_tree=annotated_doc.parse_tree,
            created_at=annotated_doc.created_at,
            updated_at=annotated_doc.updated_at,
        )

    def annotate_corpus(
        self, documents: List[AnnotatedDocument], show_progress: bool = True
    ) -> List[AnnotatedDocument]:
        """Extract relations for a corpus of annotated documents.

        Args:
            documents: List of annotated documents
            show_progress: Whether to show progress bar

        Returns:
            List of documents with relation annotations
        """
        self.logger.info(f"Extracting relations from {len(documents)} documents")

        doc_iter = tqdm(documents, desc="Extracting") if show_progress else documents

        annotated = [self.annotate_document(doc) for doc in doc_iter]

        total_relations = sum(len(doc.relations) for doc in annotated)
        self.logger.info(
            f"Relation extraction complete: {total_relations} relations "
            f"extracted from {len(documents)} documents"
        )

        return annotated


class AnnotationValidator:
    """Validate annotation quality and consistency.

    Checks for common annotation errors such as overlapping entities,
    invalid offsets, inconsistent relations, etc.
    """

    def __init__(self, strict: bool = False):
        """Initialize the annotation validator.

        Args:
            strict: Whether to raise errors or just log warnings
        """
        self.strict = strict
        self.logger = get_logger(__name__)

    def validate_entity_offsets(self, document: AnnotatedDocument) -> List[str]:
        """Validate entity offsets are within document bounds.

        Args:
            document: Annotated document to validate

        Returns:
            List of error messages
        """
        errors = []
        text_length = len(document.text)

        for entity in document.entities:
            if entity.start < 0:
                errors.append(
                    f"Entity '{entity.text}' has negative start offset: {entity.start}"
                )

            if entity.end > text_length:
                errors.append(
                    f"Entity '{entity.text}' end offset {entity.end} "
                    f"exceeds document length {text_length}"
                )

            if entity.start >= entity.end:
                errors.append(
                    f"Entity '{entity.text}' has invalid offsets: "
                    f"start={entity.start}, end={entity.end}"
                )

            # Verify text matches
            actual_text = document.text[entity.start : entity.end]
            if actual_text != entity.text:
                errors.append(
                    f"Entity text '{entity.text}' does not match "
                    f"document span '{actual_text}'"
                )

        return errors

    def validate_entity_overlaps(self, document: AnnotatedDocument) -> List[str]:
        """Check for overlapping entities.

        Args:
            document: Annotated document to validate

        Returns:
            List of error messages
        """
        errors = []
        sorted_entities = sorted(document.entities, key=lambda e: e.start)

        for i in range(len(sorted_entities) - 1):
            current = sorted_entities[i]
            next_entity = sorted_entities[i + 1]

            if current.end > next_entity.start:
                errors.append(
                    f"Overlapping entities: '{current.text}' [{current.start}:{current.end}] "
                    f"and '{next_entity.text}' [{next_entity.start}:{next_entity.end}]"
                )

        return errors

    def validate_relations(self, document: AnnotatedDocument) -> List[str]:
        """Validate relations are between valid entities.

        Args:
            document: Annotated document to validate

        Returns:
            List of error messages
        """
        errors = []

        # Create set of entity spans for quick lookup
        entity_spans = {(e.start, e.end, e.text) for e in document.entities}

        for relation in document.relations:
            # Check head entity exists
            head_span = (relation.head.start, relation.head.end, relation.head.text)
            if head_span not in entity_spans:
                errors.append(
                    f"Relation head entity not found in document: '{relation.head.text}'"
                )

            # Check tail entity exists
            tail_span = (relation.tail.start, relation.tail.end, relation.tail.text)
            if tail_span not in entity_spans:
                errors.append(
                    f"Relation tail entity not found in document: '{relation.tail.text}'"
                )

            # Check head and tail are different
            if head_span == tail_span:
                errors.append(
                    f"Relation has identical head and tail: '{relation.head.text}'"
                )

        return errors

    def validate_document(self, document: AnnotatedDocument) -> Tuple[bool, List[str]]:
        """Validate an annotated document.

        Args:
            document: Annotated document to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        all_errors = []

        # Validate entity offsets
        all_errors.extend(self.validate_entity_offsets(document))

        # Validate entity overlaps
        all_errors.extend(self.validate_entity_overlaps(document))

        # Validate relations
        all_errors.extend(self.validate_relations(document))

        is_valid = len(all_errors) == 0

        if not is_valid:
            if self.strict:
                raise ValueError(
                    f"Validation failed for document {document.id}:\n"
                    + "\n".join(all_errors)
                )
            else:
                for error in all_errors:
                    self.logger.warning(f"Document {document.id}: {error}")

        return is_valid, all_errors

    def validate_corpus(
        self, documents: List[AnnotatedDocument], show_progress: bool = True
    ) -> Dict[str, Any]:
        """Validate a corpus of annotated documents.

        Args:
            documents: List of annotated documents
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with validation statistics
        """
        self.logger.info(f"Validating {len(documents)} annotated documents")

        doc_iter = tqdm(documents, desc="Validating") if show_progress else documents

        valid_count = 0
        total_errors = 0
        error_types: Dict[str, int] = {}

        for doc in doc_iter:
            is_valid, errors = self.validate_document(doc)

            if is_valid:
                valid_count += 1
            else:
                total_errors += len(errors)
                for error in errors:
                    # Categorize errors
                    if "offset" in error.lower():
                        error_types["offset_errors"] = (
                            error_types.get("offset_errors", 0) + 1
                        )
                    elif "overlap" in error.lower():
                        error_types["overlap_errors"] = (
                            error_types.get("overlap_errors", 0) + 1
                        )
                    elif "relation" in error.lower():
                        error_types["relation_errors"] = (
                            error_types.get("relation_errors", 0) + 1
                        )
                    else:
                        error_types["other_errors"] = (
                            error_types.get("other_errors", 0) + 1
                        )

        stats = {
            "total_documents": len(documents),
            "valid_documents": valid_count,
            "invalid_documents": len(documents) - valid_count,
            "total_errors": total_errors,
            "error_types": error_types,
        }

        self.logger.info(
            f"Validation complete: {valid_count}/{len(documents)} documents valid, "
            f"{total_errors} total errors"
        )

        return stats


class GoldSetBuilder:
    """Build gold standard annotation sets with train/dev/test splits.

    Creates stratified splits for training NER and RE models.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        """Initialize the gold set builder.

        Args:
            train_ratio: Proportion of data for training
            dev_ratio: Proportion of data for development
            test_ratio: Proportion of data for testing
            random_seed: Random seed for reproducibility

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {train_ratio + dev_ratio + test_ratio}"
            )

        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.logger = get_logger(__name__)

    def split_documents(
        self, documents: List[AnnotatedDocument]
    ) -> Dict[str, List[AnnotatedDocument]]:
        """Split documents into train/dev/test sets.

        Args:
            documents: List of annotated documents

        Returns:
            Dictionary with 'train', 'dev', 'test' keys mapping to document lists
        """
        # Shuffle documents with fixed seed for reproducibility
        random.seed(self.random_seed)
        shuffled = documents.copy()
        random.shuffle(shuffled)

        # Calculate split indices
        n = len(documents)
        train_end = int(n * self.train_ratio)
        dev_end = train_end + int(n * self.dev_ratio)

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
        splits: Dict[str, List[AnnotatedDocument]],
        output_dir: Union[str, Path],
        format: str = "jsonl",
    ) -> None:
        """Save train/dev/test splits to files.

        Args:
            splits: Dictionary of split name to documents
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

        for split_name, docs in splits.items():
            output_file = output_dir / f"{split_name}.{format}"

            if format == "jsonl":
                import jsonlines

                with jsonlines.open(output_file, mode="w") as writer:
                    for doc in docs:
                        writer.write(doc.model_dump())
            elif format == "json":
                import json

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(
                        [doc.model_dump() for doc in docs],
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Saved {len(docs)} documents to {output_file}")

        self.logger.info(
            f"Consider versioning splits with DVC:\n"
            f"  dvc add {output_dir}/train.{format}\n"
            f"  dvc add {output_dir}/dev.{format}\n"
            f"  dvc add {output_dir}/test.{format}"
        )
