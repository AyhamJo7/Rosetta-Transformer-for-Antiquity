"""FastAPI application for Rosetta Transformer.

This module provides the main FastAPI application with endpoints for:
- Named Entity Recognition (NER)
- Relation Extraction (RE)
- Transliteration
- Translation
- Semantic search
- Feedback submission

The API supports batch processing, request validation, error handling,
and automatic API documentation.
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from rosetta.api.feedback import FeedbackStore
from rosetta.api.inference import InferenceEngine, ModelRegistry
from rosetta.api.search import SemanticSearchEngine
from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Current timestamp")
    models_loaded: Dict[str, bool] = Field(..., description="Status of loaded models")


class NERRequest(BaseModel):
    """Request model for Named Entity Recognition."""

    text: Union[str, List[str]] = Field(
        ..., description="Text or list of texts to analyze"
    )
    language: str = Field(default="en", description="Language code (en, ar, grc, etc.)")
    return_confidence: bool = Field(
        default=True, description="Return confidence scores"
    )
    batch_size: Optional[int] = Field(
        default=None, description="Batch size for processing"
    )

    @validator("text")
    def validate_text(cls, v):
        """Validate text input."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Text cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Text list cannot be empty")
            if not all(isinstance(t, str) and t.strip() for t in v):
                raise ValueError("All texts must be non-empty strings")
        return v


class Entity(BaseModel):  # type: ignore[misc]
    """Named entity with position and label."""

    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity label (PER, LOC, ORG, etc.)")
    start: int = Field(..., description="Start character position")
    end: int = Field(..., description="End character position")
    confidence: Optional[float] = Field(None, description="Confidence score")


class NERResponse(BaseModel):
    """Response model for Named Entity Recognition."""

    text: str = Field(..., description="Input text")
    entities: List[Entity] = Field(..., description="Extracted entities")
    language: str = Field(..., description="Language code")
    processing_time: float = Field(..., description="Processing time in seconds")


class RelationRequest(BaseModel):
    """Request model for Relation Extraction."""

    text: Union[str, List[str]] = Field(..., description="Text or list of texts")
    entities: Optional[List[List[Entity]]] = Field(
        None, description="Pre-extracted entities (optional)"
    )
    language: str = Field(default="en", description="Language code")
    return_confidence: bool = Field(
        default=True, description="Return confidence scores"
    )
    batch_size: Optional[int] = Field(
        default=None, description="Batch size for processing"
    )


class Relation(BaseModel):  # type: ignore[misc]
    """Relation between two entities."""

    head: Entity = Field(..., description="Head entity")
    tail: Entity = Field(..., description="Tail entity")
    relation: str = Field(..., description="Relation type")
    confidence: Optional[float] = Field(None, description="Confidence score")


class RelationResponse(BaseModel):
    """Response model for Relation Extraction."""

    text: str = Field(..., description="Input text")
    entities: List[Entity] = Field(..., description="Extracted entities")
    relations: List[Relation] = Field(..., description="Extracted relations")
    language: str = Field(..., description="Language code")
    processing_time: float = Field(..., description="Processing time in seconds")


class TransliterationRequest(BaseModel):
    """Request model for Transliteration."""

    text: Union[str, List[str]] = Field(..., description="Text to transliterate")
    source_script: str = Field(..., description="Source script (e.g., 'grc', 'ar')")
    target_script: str = Field(
        default="latin", description="Target script (e.g., 'latin', 'ipa')"
    )
    num_beams: int = Field(default=4, description="Number of beams for beam search")
    batch_size: Optional[int] = Field(
        default=None, description="Batch size for processing"
    )

    @validator("text")
    def validate_text(cls, v):
        """Validate text input."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Text cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Text list cannot be empty")
        return v


class TransliterationResponse(BaseModel):
    """Response model for Transliteration."""

    source_text: str = Field(..., description="Original text")
    transliteration: str = Field(..., description="Transliterated text")
    source_script: str = Field(..., description="Source script")
    target_script: str = Field(..., description="Target script")
    processing_time: float = Field(..., description="Processing time in seconds")


class TranslationRequest(BaseModel):
    """Request model for Translation."""

    text: Union[str, List[str]] = Field(..., description="Text to translate")
    source_lang: str = Field(..., description="Source language code")
    target_lang: str = Field(..., description="Target language code")
    num_beams: int = Field(default=4, description="Number of beams for beam search")
    batch_size: Optional[int] = Field(
        default=None, description="Batch size for processing"
    )

    @validator("text")
    def validate_text(cls, v):
        """Validate text input."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Text cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Text list cannot be empty")
        return v


class TranslationResponse(BaseModel):
    """Response model for Translation."""

    source_text: str = Field(..., description="Original text")
    translation: str = Field(..., description="Translated text")
    source_lang: str = Field(..., description="Source language")
    target_lang: str = Field(..., description="Target language")
    processing_time: float = Field(..., description="Processing time in seconds")


class SearchRequest(BaseModel):
    """Request model for Semantic Search."""

    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, description="Number of results to return")
    language: Optional[str] = Field(None, description="Filter by language")
    filters: Optional[Dict[str, str]] = Field(None, description="Additional filters")
    hybrid: bool = Field(
        default=True, description="Use hybrid semantic + keyword search"
    )

    @validator("query")
    def validate_query(cls, v):
        """Validate query input."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v

    @validator("top_k")
    def validate_top_k(cls, v):
        """Validate top_k parameter."""
        if v < 1 or v > 100:
            raise ValueError("top_k must be between 1 and 100")
        return v


class SearchResult(BaseModel):
    """Single search result."""

    text: str = Field(..., description="Result text")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, str] = Field(..., description="Result metadata")


class SearchResponse(BaseModel):
    """Response model for Semantic Search."""

    query: str = Field(..., description="Search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results")
    processing_time: float = Field(..., description="Processing time in seconds")


class FeedbackRequest(BaseModel):
    """Request model for Feedback submission."""

    task_type: str = Field(
        ..., description="Task type (ner, relation, transliteration, translation)"
    )
    input_text: str = Field(..., description="Original input text")
    prediction: Union[Dict, List, str] = Field(..., description="Model prediction")
    correction: Union[Dict, List, str] = Field(..., description="User correction")
    metadata: Optional[Dict[str, str]] = Field(None, description="Additional metadata")

    @validator("task_type")
    def validate_task_type(cls, v):
        """Validate task type."""
        allowed_types = ["ner", "relation", "transliteration", "translation"]
        if v not in allowed_types:
            raise ValueError(f"task_type must be one of {allowed_types}")
        return v


class FeedbackResponse(BaseModel):
    """Response model for Feedback submission."""

    success: bool = Field(..., description="Whether feedback was stored successfully")
    feedback_id: str = Field(..., description="Unique feedback ID")
    message: str = Field(..., description="Response message")


class BatchResponse(BaseModel):
    """Response model for batch operations."""

    results: List[
        Union[
            NERResponse, RelationResponse, TransliterationResponse, TranslationResponse
        ]
    ] = Field(..., description="List of individual responses")
    total_items: int = Field(..., description="Total number of items processed")
    total_processing_time: float = Field(..., description="Total processing time")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(None, description="Additional error details")


# ============================================================================
# Application Lifecycle Management
# ============================================================================

# Global instances
model_registry: Optional[ModelRegistry] = None
inference_engine: Optional[InferenceEngine] = None
search_engine: Optional[SemanticSearchEngine] = None
feedback_store: Optional[FeedbackStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.

    Handles model loading on startup and cleanup on shutdown.
    """
    global model_registry, inference_engine, search_engine, feedback_store

    logger.info("Starting Rosetta Transformer API...")

    # Initialize components
    try:
        # Initialize model registry
        model_registry = ModelRegistry()
        logger.info("Model registry initialized")

        # Initialize inference engine
        inference_engine = InferenceEngine(model_registry=model_registry)
        logger.info("Inference engine initialized")

        # Initialize search engine
        try:
            search_engine = SemanticSearchEngine()
            logger.info("Search engine initialized")
        except Exception as e:
            logger.warning(f"Search engine initialization failed: {e}")
            search_engine = None

        # Initialize feedback store
        feedback_store = FeedbackStore()
        logger.info("Feedback store initialized")

        logger.info("Rosetta Transformer API ready!")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down Rosetta Transformer API...")
    if inference_engine:
        inference_engine.cleanup()
    if search_engine:
        search_engine.cleanup()
    logger.info("Shutdown complete")


# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Rosetta Transformer API",
    description="""
    Multilingual Transformer API for Ancient and Low-Resource Texts

    This API provides state-of-the-art NLP capabilities for analyzing ancient texts:
    - Named Entity Recognition (NER)
    - Relation Extraction (RE)
    - Transliteration
    - Translation
    - Semantic Search
    - Human-in-the-loop Feedback

    Built with FastAPI, PyTorch, and Transformers.
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ============================================================================
# CORS Middleware Configuration
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # API server
        "http://localhost:8080",  # Alternative frontend
        # Add production origins as needed
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request Timing Middleware
# ============================================================================


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred",
            details={"error": str(exc)},
        ).dict(),
    )


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Rosetta Transformer API",
        "version": "0.1.0",
        "description": "Multilingual NLP for Ancient Texts",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Returns the current status of the API and loaded models.
    """
    models_loaded = {}

    if model_registry:
        models_loaded = model_registry.get_loaded_models()

    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        models_loaded=models_loaded,
    )


@app.post("/predict/ner", response_model=Union[NERResponse, BatchResponse])
async def predict_ner(request: NERRequest):
    """Named Entity Recognition endpoint.

    Extracts named entities (persons, locations, organizations, etc.)
    from ancient texts.

    Args:
        request: NER request with text and parameters

    Returns:
        NER response with extracted entities

    Raises:
        HTTPException: If inference fails
    """
    if not inference_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available",
        )

    start_time = time.time()

    try:
        # Handle batch processing
        if isinstance(request.text, list):
            results = []
            for text in request.text:
                entities = await inference_engine.predict_ner(
                    text=text,
                    language=request.language,
                    return_confidence=request.return_confidence,
                )
                results.append(
                    NERResponse(
                        text=text,
                        entities=entities,
                        language=request.language,
                        processing_time=time.time() - start_time,
                    )
                )

            return BatchResponse(
                results=results,
                total_items=len(results),
                total_processing_time=time.time() - start_time,
            )

        # Single text processing
        entities = await inference_engine.predict_ner(
            text=request.text,
            language=request.language,
            return_confidence=request.return_confidence,
        )

        return NERResponse(
            text=request.text,
            entities=entities,
            language=request.language,
            processing_time=time.time() - start_time,
        )

    except Exception as e:
        logger.error(f"NER prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NER prediction failed: {str(e)}",
        )


@app.post("/predict/relation", response_model=Union[RelationResponse, BatchResponse])
async def predict_relation(request: RelationRequest):
    """Relation Extraction endpoint.

    Identifies semantic relations between entities in ancient texts.

    Args:
        request: Relation extraction request

    Returns:
        Relation extraction response with entities and relations

    Raises:
        HTTPException: If inference fails
    """
    if not inference_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available",
        )

    start_time = time.time()

    try:
        # Handle batch processing
        if isinstance(request.text, list):
            results = []
            for i, text in enumerate(request.text):
                entities_for_text = request.entities[i] if request.entities else None

                entities, relations = await inference_engine.predict_relation(
                    text=text,
                    entities=entities_for_text,
                    language=request.language,
                    return_confidence=request.return_confidence,
                )

                results.append(
                    RelationResponse(
                        text=text,
                        entities=entities,
                        relations=relations,
                        language=request.language,
                        processing_time=time.time() - start_time,
                    )
                )

            return BatchResponse(
                results=results,
                total_items=len(results),
                total_processing_time=time.time() - start_time,
            )

        # Single text processing
        entities, relations = await inference_engine.predict_relation(
            text=request.text,
            entities=request.entities[0] if request.entities else None,
            language=request.language,
            return_confidence=request.return_confidence,
        )

        return RelationResponse(
            text=request.text,
            entities=entities,
            relations=relations,
            language=request.language,
            processing_time=time.time() - start_time,
        )

    except Exception as e:
        logger.error(f"Relation extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Relation extraction failed: {str(e)}",
        )


@app.post(
    "/predict/transliterate",
    response_model=Union[TransliterationResponse, BatchResponse],
)
async def predict_transliterate(request: TransliterationRequest):
    """Transliteration endpoint.

    Converts text from one script to another (e.g., Ancient Greek to Latin).

    Args:
        request: Transliteration request

    Returns:
        Transliteration response

    Raises:
        HTTPException: If inference fails
    """
    if not inference_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available",
        )

    start_time = time.time()

    try:
        # Handle batch processing
        if isinstance(request.text, list):
            results = []
            transliterations = await inference_engine.transliterate(
                texts=request.text,
                source_script=request.source_script,
                target_script=request.target_script,
                num_beams=request.num_beams,
                batch_size=request.batch_size,
            )

            for source_text, trans_text in zip(request.text, transliterations):
                results.append(
                    TransliterationResponse(
                        source_text=source_text,
                        transliteration=trans_text,
                        source_script=request.source_script,
                        target_script=request.target_script,
                        processing_time=time.time() - start_time,
                    )
                )

            return BatchResponse(
                results=results,
                total_items=len(results),
                total_processing_time=time.time() - start_time,
            )

        # Single text processing
        transliterations = await inference_engine.transliterate(
            texts=[request.text],
            source_script=request.source_script,
            target_script=request.target_script,
            num_beams=request.num_beams,
            batch_size=1,
        )

        return TransliterationResponse(
            source_text=request.text,
            transliteration=transliterations[0],
            source_script=request.source_script,
            target_script=request.target_script,
            processing_time=time.time() - start_time,
        )

    except Exception as e:
        logger.error(f"Transliteration failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transliteration failed: {str(e)}",
        )


@app.post(
    "/predict/translate", response_model=Union[TranslationResponse, BatchResponse]
)
async def predict_translate(request: TranslationRequest):
    """Translation endpoint.

    Translates text from one language to another.

    Args:
        request: Translation request

    Returns:
        Translation response

    Raises:
        HTTPException: If inference fails
    """
    if not inference_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available",
        )

    start_time = time.time()

    try:
        # Handle batch processing
        if isinstance(request.text, list):
            results = []
            translations = await inference_engine.translate(
                texts=request.text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                num_beams=request.num_beams,
                batch_size=request.batch_size,
            )

            for source_text, trans_text in zip(request.text, translations):
                results.append(
                    TranslationResponse(
                        source_text=source_text,
                        translation=trans_text,
                        source_lang=request.source_lang,
                        target_lang=request.target_lang,
                        processing_time=time.time() - start_time,
                    )
                )

            return BatchResponse(
                results=results,
                total_items=len(results),
                total_processing_time=time.time() - start_time,
            )

        # Single text processing
        translations = await inference_engine.translate(
            texts=[request.text],
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            num_beams=request.num_beams,
            batch_size=1,
        )

        return TranslationResponse(
            source_text=request.text,
            translation=translations[0],
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            processing_time=time.time() - start_time,
        )

    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}",
        )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semantic search endpoint.

    Searches the corpus using semantic similarity and optionally
    keyword matching (hybrid search).

    Args:
        request: Search request with query and parameters

    Returns:
        Search response with ranked results

    Raises:
        HTTPException: If search fails or search engine is unavailable
    """
    if not search_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search engine not available",
        )

    start_time = time.time()

    try:
        results = await search_engine.search(
            query=request.query,
            top_k=request.top_k,
            language=request.language,
            filters=request.filters,
            hybrid=request.hybrid,
        )

        return SearchResponse(
            query=request.query,
            results=results,
            total_found=len(results),
            processing_time=time.time() - start_time,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Feedback submission endpoint.

    Allows users to submit corrections and feedback on model predictions
    for continuous improvement and active learning.

    Args:
        request: Feedback request with prediction and correction

    Returns:
        Feedback response with confirmation

    Raises:
        HTTPException: If feedback submission fails
    """
    if not feedback_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback store not available",
        )

    try:
        feedback_id = await feedback_store.add_feedback(
            task_type=request.task_type,
            input_text=request.input_text,
            prediction=request.prediction,
            correction=request.correction,
            metadata=request.metadata,
        )

        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message="Feedback submitted successfully",
        )

    except Exception as e:
        logger.error(f"Feedback submission failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}",
        )


@app.get("/feedback/stats", response_model=Dict[str, int])
async def get_feedback_stats():
    """Get feedback statistics.

    Returns:
        Dictionary with feedback counts by task type
    """
    if not feedback_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback store not available",
        )

    try:
        stats = await feedback_store.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feedback stats: {str(e)}",
        )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "rosetta.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
