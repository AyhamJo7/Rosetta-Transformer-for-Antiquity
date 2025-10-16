"""Semantic search engine for Rosetta Transformer.

This module provides:
- EmbeddingGenerator: Generate embeddings using sentence-transformers
- FAISSIndex: FAISS-based vector similarity search
- SemanticSearchEngine: High-level search interface
- Hybrid search: Combine semantic and keyword-based search
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Embedding Generator
# ============================================================================


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers.

    Uses multilingual models to support ancient and modern languages.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """Initialize embedding generator.

        Args:
            model_name: Name of sentence-transformer model
            device: Device to use (cuda/cpu, auto-detected if None)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size

        logger.info(f"Loading embedding model: {model_name}")

        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"Embedding model loaded. Dimension: {self.embedding_dim}, "
            f"Device: {self.model.device}"
        )

    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size (uses default if None)
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if batch_size is None:
            batch_size = self.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return embeddings

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Encode single query.

        Args:
            query: Query text
            normalize: Normalize embedding

        Returns:
            Query embedding (embedding_dim,)
        """
        embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=normalize
        )[0]

        return embedding


# ============================================================================
# FAISS Index
# ============================================================================


class FAISSIndex:
    """FAISS-based vector similarity search.

    Supports different index types for speed/accuracy tradeoffs.
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        metric: str = "cosine",
    ):
        """Initialize FAISS index.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index (flat, ivf, hnsw)
            metric: Distance metric (cosine, l2)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric

        # Create index
        self.index = self._create_index()

        # Store document metadata
        self.documents: List[Dict[str, Any]] = []

        logger.info(
            f"FAISS index created: type={index_type}, "
            f"metric={metric}, dim={embedding_dim}"
        )

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on type.

        Returns:
            FAISS index
        """
        if self.metric == "cosine":
            # For cosine similarity, use inner product with normalized vectors
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.metric == "l2":
            index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Wrap in index type
        if self.index_type == "flat":
            # Already flat
            pass
        elif self.index_type == "ivf":
            # IVF for large datasets
            nlist = 100  # Number of clusters
            quantizer = index
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif self.index_type == "hnsw":
            # HNSW for fast approximate search
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]],
    ) -> None:
        """Add embeddings and documents to index.

        Args:
            embeddings: Embeddings array (n_docs, embedding_dim)
            documents: List of document metadata
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)

        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        # Add to index
        self.index.add(embeddings)
        self.documents.extend(documents)

        logger.info(f"Added {len(embeddings)} documents to index")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding (embedding_dim,)
            top_k: Number of results to return

        Returns:
            Tuple of (distances, indices)
        """
        # Ensure query is 2D and float32
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        return distances[0], indices[0]

    def get_documents(self, indices: np.ndarray) -> List[Dict[str, Any]]:
        """Get documents by indices.

        Args:
            indices: Array of document indices

        Returns:
            List of documents
        """
        return [self.documents[i] for i in indices if 0 <= i < len(self.documents)]

    def save(self, index_path: str, metadata_path: str) -> None:
        """Save index and metadata to disk.

        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save document metadata
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save metadata
        with open(metadata_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "embedding_dim": self.embedding_dim,
                    "index_type": self.index_type,
                    "metric": self.metric,
                },
                f,
            )

        logger.info(f"Index saved to {index_path}")

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FAISSIndex":
        """Load index and metadata from disk.

        Args:
            index_path: Path to FAISS index
            metadata_path: Path to document metadata

        Returns:
            Loaded FAISSIndex
        """
        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls(
            embedding_dim=metadata["embedding_dim"],
            index_type=metadata["index_type"],
            metric=metadata["metric"],
        )

        # Load FAISS index
        instance.index = faiss.read_index(index_path)
        instance.documents = metadata["documents"]

        logger.info(f"Index loaded from {index_path}")

        return instance

    def __len__(self) -> int:
        """Get number of documents in index."""
        return self.index.ntotal


# ============================================================================
# Semantic Search Engine
# ============================================================================


class SemanticSearchEngine:
    """High-level semantic search engine.

    Combines embedding generation, indexing, and retrieval.
    Supports hybrid search with keyword matching.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        index_dir: Optional[str] = None,
        index_type: str = "flat",
        metric: str = "cosine",
    ):
        """Initialize semantic search engine.

        Args:
            embedding_model: Name of embedding model
            index_dir: Directory to load/save index
            index_type: Type of FAISS index
            metric: Distance metric
        """
        self.index_dir = Path(index_dir) if index_dir else Path("data/search_index")
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)

        # Initialize or load index
        index_path = self.index_dir / "faiss.index"
        metadata_path = self.index_dir / "metadata.pkl"

        if index_path.exists() and metadata_path.exists():
            logger.info("Loading existing search index...")
            self.index = FAISSIndex.load(str(index_path), str(metadata_path))
        else:
            logger.info("Creating new search index...")
            self.index = FAISSIndex(
                embedding_dim=self.embedding_generator.embedding_dim,
                index_type=index_type,
                metric=metric,
            )

        logger.info(f"Search engine initialized with {len(self.index)} documents")

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> None:
        """Index documents for search.

        Args:
            documents: List of documents with text and metadata
            text_field: Field name containing text to index
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        """
        logger.info(f"Indexing {len(documents)} documents...")

        # Extract texts
        texts = [doc[text_field] for doc in documents]

        # Generate embeddings
        embeddings = self.embedding_generator.encode(
            texts, batch_size=batch_size, show_progress=show_progress
        )

        # Add to index
        self.index.add(embeddings, documents)

        # Save index
        self.save()

        logger.info(f"Indexing complete. Total documents: {len(self.index)}")

    async def search(
        self,
        query: str,
        top_k: int = 10,
        language: Optional[str] = None,
        filters: Optional[Dict[str, str]] = None,
        hybrid: bool = True,
        keyword_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents.

        Args:
            query: Search query
            top_k: Number of results to return
            language: Filter by language
            filters: Additional metadata filters
            hybrid: Use hybrid semantic + keyword search
            keyword_weight: Weight for keyword scores (0-1)

        Returns:
            List of search results with scores
        """
        if len(self.index) == 0:
            logger.warning("Search index is empty")
            return []

        # Generate query embedding
        query_embedding = self.embedding_generator.encode_query(query)

        # Semantic search
        distances, indices = self.index.search(query_embedding, top_k=top_k * 2)

        # Get documents
        results = []
        for distance, idx in zip(distances, indices):
            if idx < 0:  # FAISS returns -1 for missing results
                continue

            doc = self.index.documents[idx]

            # Apply filters
            if language and doc.get("language") != language:
                continue

            if filters:
                if not all(doc.get(k) == v for k, v in filters.items()):
                    continue

            # Calculate score
            semantic_score = float(distance)  # Cosine similarity

            # Add keyword score if hybrid
            if hybrid:
                keyword_score = self._keyword_score(query, doc.get("text", ""))
                final_score = (
                    1 - keyword_weight
                ) * semantic_score + keyword_weight * keyword_score
            else:
                final_score = semantic_score

            result = {
                "text": doc.get("text", ""),
                "score": final_score,
                "metadata": {k: v for k, v in doc.items() if k != "text"},
            }
            results.append(result)

        # Sort by score (descending) and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        return results

    def _keyword_score(self, query: str, text: str) -> float:
        """Calculate keyword-based relevance score.

        Args:
            query: Search query
            text: Document text

        Returns:
            Keyword score (0-1)
        """
        # Simple keyword matching (can be improved with BM25)
        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())

        if not query_terms:
            return 0.0

        # Jaccard similarity
        intersection = query_terms & text_terms
        union = query_terms | text_terms

        return len(intersection) / len(union) if union else 0.0

    def add_document(self, document: Dict[str, Any], text_field: str = "text") -> None:
        """Add single document to index.

        Args:
            document: Document dict with text and metadata
            text_field: Field name containing text
        """
        text = document[text_field]
        embedding = self.embedding_generator.encode([text])
        self.index.add(embedding, [document])

    def save(self) -> None:
        """Save index to disk."""
        index_path = self.index_dir / "faiss.index"
        metadata_path = self.index_dir / "metadata.pkl"

        self.index.save(str(index_path), str(metadata_path))
        logger.info(f"Search index saved to {self.index_dir}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up search engine...")
        # Save before cleanup
        self.save()
        logger.info("Search engine cleanup complete")


# ============================================================================
# Helper Functions
# ============================================================================


def build_search_index_from_corpus(
    corpus_path: str,
    output_dir: str,
    text_field: str = "text",
    batch_size: int = 32,
) -> SemanticSearchEngine:
    """Build search index from corpus file.

    Args:
        corpus_path: Path to corpus JSONL file
        output_dir: Output directory for index
        text_field: Field containing text to index
        batch_size: Batch size for encoding

    Returns:
        Initialized SemanticSearchEngine
    """
    logger.info(f"Building search index from {corpus_path}")

    # Load corpus
    documents = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc)

    # Initialize search engine
    search_engine = SemanticSearchEngine(index_dir=output_dir)

    # Index documents
    search_engine.index_documents(
        documents, text_field=text_field, batch_size=batch_size
    )

    return search_engine


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/search_index"

        engine = build_search_index_from_corpus(corpus_path, output_dir)
        print(f"Index built successfully with {len(engine.index)} documents")
    else:
        print("Usage: python search.py <corpus_path> [output_dir]")
