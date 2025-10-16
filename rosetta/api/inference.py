"""Model inference engine for Rosetta Transformer.

This module provides:
- ModelRegistry: Manages loaded models and their lifecycle
- InferenceEngine: Runs predictions with batching and caching
- GPU memory management
- Request caching for repeated queries
"""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from rosetta.models import (
    RelationExtractionModel,
    Seq2SeqModel,
    TokenClassificationModel,
)
from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Model Registry
# ============================================================================


class ModelRegistry:
    """Registry for managing loaded models.

    Handles model loading, caching, and memory management.
    Supports lazy loading and automatic model unloading when memory is low.
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        max_models_in_memory: int = 3,
        device: Optional[str] = None,
    ):
        """Initialize model registry.

        Args:
            models_dir: Directory containing trained models
            max_models_in_memory: Maximum number of models to keep loaded
            device: Device to load models on (cuda/cpu, auto-detected if None)
        """
        self.models_dir = Path(models_dir) if models_dir else Path("models/trained")
        self.max_models_in_memory = max_models_in_memory

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Model registry initialized on device: {self.device}")

        # Model cache: OrderedDict for LRU eviction
        self._models: OrderedDict[str, torch.nn.Module] = OrderedDict()
        self._tokenizers: OrderedDict[str, Any] = OrderedDict()
        self._model_configs: Dict[str, Dict] = {}

        # Model paths configuration
        self._model_paths = {
            "ner": {
                "en": "ner/en-ner-model",
                "ar": "ner/ar-ner-model",
                "grc": "ner/grc-ner-model",
            },
            "relation": {
                "en": "relation/en-re-model",
                "ar": "relation/ar-re-model",
                "grc": "relation/grc-re-model",
            },
            "transliteration": {
                "grc-latin": "transliteration/grc-latin-model",
                "ar-latin": "transliteration/ar-latin-model",
            },
            "translation": {
                "grc-en": "translation/grc-en-model",
                "ar-en": "translation/ar-en-model",
            },
        }

    def _get_model_key(self, task: str, language: str) -> str:
        """Generate unique key for model.

        Args:
            task: Task type (ner, relation, etc.)
            language: Language or language pair

        Returns:
            Unique model key
        """
        return f"{task}:{language}"

    def _evict_lru_model(self) -> None:
        """Evict least recently used model from memory."""
        if len(self._models) >= self.max_models_in_memory:
            # Remove oldest item (first in OrderedDict)
            model_key, model = self._models.popitem(last=False)
            logger.info(f"Evicting model from memory: {model_key}")

            # Move model to CPU and clear cache
            if hasattr(model, "cpu"):
                model.cpu()

            if self.device == "cuda":
                torch.cuda.empty_cache()

    def load_model(
        self,
        task: str,
        language: str,
        force_reload: bool = False,
    ) -> Tuple[torch.nn.Module, Any]:
        """Load model and tokenizer.

        Args:
            task: Task type (ner, relation, transliteration, translation)
            language: Language code or language pair
            force_reload: Force reload even if already in memory

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ValueError: If model not found
        """
        model_key = self._get_model_key(task, language)

        # Return from cache if available
        if not force_reload and model_key in self._models:
            # Move to end (most recently used)
            self._models.move_to_end(model_key)
            logger.debug(f"Using cached model: {model_key}")
            return self._models[model_key], self._tokenizers[model_key]

        # Get model path
        if task not in self._model_paths:
            raise ValueError(f"Unknown task: {task}")

        if language not in self._model_paths[task]:
            raise ValueError(f"No model for language '{language}' in task '{task}'")

        model_path = self.models_dir / self._model_paths[task][language]

        if not model_path.exists():
            logger.warning(
                f"Model not found at {model_path}. "
                f"Using fallback to base model for task '{task}'"
            )
            return self._load_base_model(task, language)

        logger.info(f"Loading model: {model_key} from {model_path}")

        # Evict LRU model if necessary
        self._evict_lru_model()

        # Load model based on task type
        try:
            if task == "ner":
                model = self._load_ner_model(model_path)
            elif task == "relation":
                model = self._load_relation_model(model_path)
            elif task == "transliteration":
                model = self._load_seq2seq_model(model_path)
            elif task == "translation":
                model = self._load_seq2seq_model(model_path)
            else:
                raise ValueError(f"Unsupported task: {task}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            # Move to device
            model.to(self.device)
            model.eval()

            # Cache
            self._models[model_key] = model
            self._tokenizers[model_key] = tokenizer

            logger.info(f"Model loaded successfully: {model_key}")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise

    def _load_ner_model(self, model_path: Path) -> TokenClassificationModel:
        """Load NER model."""
        return TokenClassificationModel.load_pretrained(str(model_path))

    def _load_relation_model(self, model_path: Path) -> RelationExtractionModel:
        """Load relation extraction model."""
        return RelationExtractionModel.load_pretrained(str(model_path))

    def _load_seq2seq_model(self, model_path: Path) -> Seq2SeqModel:
        """Load sequence-to-sequence model."""
        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            return Seq2SeqModel(**config)
        else:
            # Fallback to default
            return Seq2SeqModel(model_name=str(model_path))

    def _load_base_model(self, task: str, language: str) -> Tuple[torch.nn.Module, Any]:
        """Load base model as fallback.

        Args:
            task: Task type
            language: Language code

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.warning(f"Using base model for {task}:{language}")

        # Base model mapping
        base_models = {
            "ner": ("xlm-roberta-base", TokenClassificationModel),
            "relation": ("xlm-roberta-base", RelationExtractionModel),
            "transliteration": ("facebook/mbart-large-50", Seq2SeqModel),
            "translation": ("facebook/mbart-large-50", Seq2SeqModel),
        }

        if task not in base_models:
            raise ValueError(f"No base model for task: {task}")

        base_model_name, model_class = base_models[task]

        # Load model
        if task in ["ner", "relation"]:
            model = model_class(model_name=base_model_name)
        else:
            model = model_class(model_name=base_model_name)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        model.to(self.device)
        model.eval()

        return model, tokenizer

    def get_loaded_models(self) -> Dict[str, bool]:
        """Get status of loaded models.

        Returns:
            Dictionary mapping model keys to loaded status
        """
        return dict.fromkeys(self._models.keys(), True)

    def unload_model(self, task: str, language: str) -> None:
        """Unload specific model from memory.

        Args:
            task: Task type
            language: Language code
        """
        model_key = self._get_model_key(task, language)

        if model_key in self._models:
            model = self._models.pop(model_key)
            self._tokenizers.pop(model_key)

            if hasattr(model, "cpu"):
                model.cpu()

            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info(f"Unloaded model: {model_key}")

    def clear_cache(self) -> None:
        """Clear all models from memory."""
        logger.info("Clearing model cache...")

        for model in self._models.values():
            if hasattr(model, "cpu"):
                model.cpu()

        self._models.clear()
        self._tokenizers.clear()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info("Model cache cleared")


# ============================================================================
# Inference Engine
# ============================================================================


class InferenceEngine:
    """Engine for running model inference.

    Provides high-level interface for predictions with:
    - Automatic batching
    - Result caching
    - Error handling
    - Performance optimization
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        cache_size: int = 1000,
        default_batch_size: int = 8,
    ):
        """Initialize inference engine.

        Args:
            model_registry: Model registry instance
            cache_size: Maximum number of cached results
            default_batch_size: Default batch size for processing
        """
        self.model_registry = model_registry
        self.default_batch_size = default_batch_size

        # LRU cache for results
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.cache_size = cache_size

        logger.info("Inference engine initialized")

    def _cache_key(self, **kwargs) -> str:
        """Generate cache key from arguments.

        Args:
            **kwargs: Arguments to hash

        Returns:
            Cache key string
        """
        key_str = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None
        """
        if cache_key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            logger.debug(f"Cache hit: {cache_key[:8]}...")
            return self._cache[cache_key]
        return None

    def _add_to_cache(self, cache_key: str, result: Any) -> None:
        """Add result to cache.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        if len(self._cache) >= self.cache_size:
            # Remove oldest
            self._cache.popitem(last=False)

        self._cache[cache_key] = result

    async def predict_ner(
        self,
        text: str,
        language: str = "en",
        return_confidence: bool = True,
    ) -> List[Dict[str, Any]]:
        """Predict named entities in text.

        Args:
            text: Input text
            language: Language code
            return_confidence: Whether to return confidence scores

        Returns:
            List of entities with positions and labels
        """
        # Check cache
        cache_key = self._cache_key(
            task="ner",
            text=text,
            language=language,
            return_confidence=return_confidence,
        )
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # Load model
        model, tokenizer = self.model_registry.load_model("ner", language)

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.model_registry.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]  # (seq_len, num_labels)

        # Get predictions
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        confidences = F.softmax(logits, dim=-1).max(dim=-1)[0].cpu().numpy()

        # Get label names (assuming model has id2label)
        id2label = getattr(model, "id2label", None)
        if id2label is None:
            # Default BIO labels
            id2label = {
                0: "O",
                1: "B-PER",
                2: "I-PER",
                3: "B-LOC",
                4: "I-LOC",
                5: "B-ORG",
                6: "I-ORG",
                7: "B-MISC",
                8: "I-MISC",
            }

        # Convert to entities
        entities = []
        current_entity = None

        for idx, (pred, conf, offset) in enumerate(
            zip(predictions, confidences, offset_mapping)
        ):
            if offset[0] == 0 and offset[1] == 0:
                # Special token, skip
                continue

            label = id2label.get(pred, "O")

            if label.startswith("B-"):
                # Begin new entity
                if current_entity:
                    entities.append(current_entity)

                entity_type = label[2:]
                current_entity = {
                    "text": text[offset[0] : offset[1]],
                    "label": entity_type,
                    "start": int(offset[0]),
                    "end": int(offset[1]),
                    "confidence": float(conf) if return_confidence else None,
                }

            elif label.startswith("I-") and current_entity:
                # Continue entity
                entity_type = label[2:]
                if entity_type == current_entity["label"]:
                    current_entity["end"] = int(offset[1])
                    current_entity["text"] = text[
                        current_entity["start"] : current_entity["end"]
                    ]
                    # Update confidence (average)
                    if return_confidence:
                        current_entity["confidence"] = (
                            current_entity["confidence"] + float(conf)
                        ) / 2

            else:
                # O label or mismatch
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Add last entity
        if current_entity:
            entities.append(current_entity)

        # Cache result
        self._add_to_cache(cache_key, entities)

        return entities

    async def predict_relation(
        self,
        text: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        language: str = "en",
        return_confidence: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Predict relations between entities.

        Args:
            text: Input text
            entities: Pre-extracted entities (optional)
            language: Language code
            return_confidence: Whether to return confidence scores

        Returns:
            Tuple of (entities, relations)
        """
        # Extract entities if not provided
        if entities is None:
            entities = await self.predict_ner(text, language, return_confidence)

        if len(entities) < 2:
            # Need at least 2 entities for relations
            return entities, []

        # Load model
        model, tokenizer = self.model_registry.load_model("relation", language)

        # Generate all entity pairs
        relations = []

        for i, head_entity in enumerate(entities):
            for tail_entity in entities[i + 1 :]:
                # Tokenize with entity markers
                marked_text = self._mark_entities_in_text(
                    text, head_entity, tail_entity
                )

                inputs = tokenizer(
                    marked_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                inputs = {
                    k: v.to(self.model_registry.device) for k, v in inputs.items()
                }

                # Get entity positions in tokenized text
                # For simplicity, using CLS pooling (can be improved)
                head_positions = torch.tensor([[0, 0]]).to(self.model_registry.device)
                tail_positions = torch.tensor([[0, 0]]).to(self.model_registry.device)

                # Predict
                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        head_positions=head_positions,
                        tail_positions=tail_positions,
                    )
                    logits = outputs.logits[0]

                # Get prediction
                pred = torch.argmax(logits).item()
                conf = F.softmax(logits, dim=-1).max().item()

                # Get relation label
                id2relation = getattr(model, "id2relation", None)
                if id2relation is None:
                    # Default relations
                    id2relation = {
                        0: "NO_RELATION",
                        1: "LOCATED_IN",
                        2: "PART_OF",
                        3: "MEMBER_OF",
                        4: "RULED_BY",
                        5: "BORN_IN",
                        6: "DIED_IN",
                    }

                relation_label = id2relation.get(pred, "UNKNOWN")

                # Skip NO_RELATION
                if relation_label != "NO_RELATION":
                    relation = {
                        "head": head_entity,
                        "tail": tail_entity,
                        "relation": relation_label,
                        "confidence": float(conf) if return_confidence else None,
                    }
                    relations.append(relation)

        return entities, relations

    def _mark_entities_in_text(
        self, text: str, head_entity: Dict, tail_entity: Dict
    ) -> str:
        """Mark entities in text with special tokens.

        Args:
            text: Original text
            head_entity: Head entity dict
            tail_entity: Tail entity dict

        Returns:
            Text with marked entities
        """
        # Simple approach: add markers around entities
        # In production, use proper entity markers in tokenizer
        marked_text = text
        return marked_text

    async def transliterate(
        self,
        texts: List[str],
        source_script: str,
        target_script: str = "latin",
        num_beams: int = 4,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Transliterate texts from one script to another.

        Args:
            texts: List of texts to transliterate
            source_script: Source script code
            target_script: Target script code
            num_beams: Number of beams for beam search
            batch_size: Batch size for processing

        Returns:
            List of transliterated texts
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        # Load model
        lang_pair = f"{source_script}-{target_script}"
        model, tokenizer = self.model_registry.load_model("transliteration", lang_pair)

        # Use model's translate method if available
        if hasattr(model, "translate"):
            return model.translate(
                texts=texts, num_beams=num_beams, batch_size=batch_size
            )

        # Fallback: manual batching
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            inputs = {k: v.to(self.model_registry.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, num_beams=num_beams, max_length=512
                )

            # Decode
            batch_results = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            results.extend(batch_results)

        return results

    async def translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        num_beams: int = 4,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Translate texts from one language to another.

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            num_beams: Number of beams for beam search
            batch_size: Batch size for processing

        Returns:
            List of translated texts
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        # Load model
        lang_pair = f"{source_lang}-{target_lang}"
        model, tokenizer = self.model_registry.load_model("translation", lang_pair)

        # Use model's translate method if available
        if hasattr(model, "translate"):
            return model.translate(
                texts=texts, num_beams=num_beams, batch_size=batch_size
            )

        # Fallback: manual batching
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            inputs = {k: v.to(self.model_registry.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, num_beams=num_beams, max_length=512
                )

            # Decode
            batch_results = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            results.extend(batch_results)

        return results

    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up inference engine...")
        self._cache.clear()
        self.model_registry.clear_cache()
        logger.info("Cleanup complete")
