"""
Cybersecurity Knowledge Base — Embedding Layer.

Converts KnowledgeChunk text into dense numerical vectors suitable for
semantic indexing and retrieval in the next pipeline stage (FAISS).

Pipeline position:
    KnowledgeChunk  (from chunking.py)
        ↓
    EmbeddingModel  (this module)
        ↓
    EmbeddedChunks  (chunks + float32 NumPy matrix)
        ↓
    NEXT STEP: FAISS Vector Store

Design Principles:
    1. One job: text → vector. No FAISS, no retrieval, no LLM, no threats.
    2. Model loaded once per EmbeddingModel instance (no per-call reloading).
    3. Batch encoding (not chunk-by-chunk) for efficiency.
    4. L2-normalised float32 vectors for cosine-similarity via FAISS dot product.
    5. Strict ordering: embeddings[i] ↔ chunks[i] always.
    6. Injectable backend protocol for unit testing without network I/O.
    7. Empty/whitespace-only text raises ValueError; empty chunk list returns
       a zero-row 2-D NumPy array of shape (0, embedding_dimension).

Default model:
    sentence-transformers/all-MiniLM-L6-v2
    - 384-dimensional embeddings
    - Apache 2.0 licence
    - No external API key required
    - Cached locally after first download
    - Excellent quality/size tradeoff for a beginner-friendly MVP

Normalisation:
    Embeddings are L2-normalised before being returned so that cosine
    similarity equals the dot product of two vectors.  This is the
    expected input format for FAISS IndexFlatIP (inner-product index).
    Normalisation is applied explicitly and documented here so the FAISS
    step can rely on it without re-normalising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from app.rag.schemas import KnowledgeChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default sentence-transformers model identifier.
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

#: Default batch size for encoding multiple chunks.
#: sentence-transformers handles variable-length inputs well at this size.
DEFAULT_BATCH_SIZE: int = 64


# ---------------------------------------------------------------------------
# Backend protocol — allows injection of a mock for unit tests
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingBackend(Protocol):
    """
    Protocol (interface) that any embedding backend must satisfy.

    The real backend wraps SentenceTransformer.  Unit tests can inject a
    lightweight stub that never touches the network.

    Responsibilities:
        - encode() a list of strings into a 2-D float32 NumPy array.
        - expose the fixed output dimensionality.
    """

    @property
    def embedding_dim(self) -> int:
        """Fixed output vector dimension (e.g. 384 for all-MiniLM-L6-v2)."""
        ...  # pragma: no cover

    def encode(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> npt.NDArray[np.float32]:
        """
        Encode a list of texts into a 2-D float32 array.

        Args:
            texts:      Non-empty list of non-empty strings.
            batch_size: Number of texts per encoding batch.

        Returns:
            NumPy array of shape (len(texts), embedding_dim), dtype float32.
            Row i corresponds to texts[i].
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Real backend — wraps sentence-transformers
# ---------------------------------------------------------------------------

class SentenceTransformerBackend:
    """
    Production embedding backend backed by the sentence-transformers library.

    The underlying SentenceTransformer is loaded once in __init__ and reused
    across all subsequent encode() calls on this instance.

    Args:
        model_name: HuggingFace model identifier or local path.
                    Defaults to DEFAULT_MODEL_NAME.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        # Import here so the library is only required when this backend is used.
        # This keeps tests that use a mock backend free of heavy imports.
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerBackend. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        logger.info("Embedding model loaded successfully: %s", model_name)

    @property
    def embedding_dim(self) -> int:
        """Return the fixed output dimension of the loaded model."""
        return int(self._model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> npt.NDArray[np.float32]:
        """
        Encode texts in batches using the loaded SentenceTransformer.

        Args:
            texts:      Non-empty list of non-empty strings.
            batch_size: Number of texts per encoding batch.

        Returns:
            Float32 NumPy array of shape (len(texts), embedding_dim).
        """
        # sentence-transformers returns a numpy array by default (or torch
        # tensor depending on version).  We explicitly convert to float32.
        raw = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.array(raw, dtype=np.float32)


# ---------------------------------------------------------------------------
# EmbeddedChunks — result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddedChunks:
    """
    Immutable result container pairing KnowledgeChunks with their embeddings.

    Ordering invariant:
        chunks[i]  <->  embeddings[i, :]  for all i in range(len(chunks))

    Attributes:
        chunks:     Ordered list of KnowledgeChunk objects (input order preserved).
        embeddings: Float32 NumPy array of shape (N, embedding_dim), where
                    N == len(chunks) and embedding_dim is the model's output size.
                    Rows are L2-normalised for cosine-similarity via dot product.
        model_name: Identifier of the model used to produce the embeddings.
    """

    chunks: List[KnowledgeChunk]
    embeddings: npt.NDArray[np.float32]
    model_name: str

    def __post_init__(self) -> None:
        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(
                f"EmbeddedChunks invariant violated: "
                f"len(chunks)={len(self.chunks)} != "
                f"embeddings.shape[0]={self.embeddings.shape[0]}"
            )

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension inferred from the stored matrix."""
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0


# ---------------------------------------------------------------------------
# EmbeddingModel — public API
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Converts KnowledgeChunk text into dense, L2-normalised float32 vectors.

    The backend (real or mock) is loaded once during construction and reused
    for all subsequent embed_text() / embed_chunks() calls on this instance.

    Args:
        model_name:  HuggingFace model ID or local path.
                     Defaults to DEFAULT_MODEL_NAME.
        batch_size:  Number of texts sent to the backend per forward pass.
                     Defaults to DEFAULT_BATCH_SIZE.
        backend:     Optional pre-built EmbeddingBackend.  When provided,
                     model_name is stored but no new backend is constructed.
                     Intended for unit testing with mock/stub backends.

    Example::

        embedder = EmbeddingModel()

        # Single text
        vector = embedder.embed_text("SQL injection allows bypassing authentication.")
        # -> numpy array of shape (384,), dtype float32, L2-normalised

        # Multiple chunks (from chunking.py)
        result = embedder.embed_chunks(chunks)
        # result.embeddings.shape == (len(chunks), 384)
        # result.chunks[i]  <->  result.embeddings[i]

    Empty-input behaviour:
        - embed_text("") raises ValueError.
        - embed_text("   ") raises ValueError.
        - embed_chunks([]) returns EmbeddedChunks with shape (0, embedding_dim).

    Normalisation:
        All returned vectors are L2-normalised.  For a unit vector u,
        cosine_similarity(u, v) == dot(u, v).  FAISS IndexFlatIP exploits this
        to perform cosine-similarity search via inner products.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size

        if backend is not None:
            # Injected backend (e.g. mock for unit tests).
            self._backend: EmbeddingBackend = backend
        else:
            # Production path: load sentence-transformers.
            self._backend = SentenceTransformerBackend(model_name)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Model identifier used by this instance."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Fixed output dimension of the embedding vectors (e.g. 384)."""
        return self._backend.embedding_dim

    # ------------------------------------------------------------------
    # Core embedding methods
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> npt.NDArray[np.float32]:
        """
        Embed a single text string into an L2-normalised float32 vector.

        Args:
            text: Non-empty, non-whitespace-only string to embed.

        Returns:
            1-D NumPy array of shape (embedding_dim,), dtype float32,
            L2-normalised (unit vector).

        Raises:
            ValueError: If text is empty or contains only whitespace.

        Example::

            vector = embedder.embed_text("OWASP Top 10: Broken Access Control")
            # vector.shape == (384,)
            # np.linalg.norm(vector) approx 1.0
        """
        if not text or not text.strip():
            raise ValueError(
                "embed_text() requires a non-empty, non-whitespace-only string. "
                f"Got: {text!r}"
            )

        matrix = self._backend.encode([text], batch_size=self._batch_size)
        vector = matrix[0]  # shape: (embedding_dim,)
        return self._normalise(vector.reshape(1, -1))[0]

    def embed_chunks(
        self,
        chunks: List[KnowledgeChunk],
    ) -> "EmbeddedChunks":
        """
        Embed a list of KnowledgeChunks in batch order.

        Ordering invariant: embeddings[i] corresponds to chunks[i].
        The input list order is preserved exactly — no sorting is performed.

        Args:
            chunks: List of KnowledgeChunk objects (may be empty).

        Returns:
            EmbeddedChunks containing:
                - chunks:     same list, same order as input.
                - embeddings: float32 NumPy array of shape (N, embedding_dim)
                              where N == len(chunks).  L2-normalised rows.
                - model_name: identifier of the model used.

            When chunks is empty, embeddings has shape (0, embedding_dim).

        Example::

            result = embedder.embed_chunks(chunks)
            assert result.embeddings.shape == (len(chunks), embedder.embedding_dim)
            # result.chunks[i]  <->  result.embeddings[i, :]
        """
        if not chunks:
            # Return a valid empty 2-D array with the correct second dimension.
            empty = np.empty((0, self.embedding_dim), dtype=np.float32)
            return EmbeddedChunks(
                chunks=[],
                embeddings=empty,
                model_name=self._model_name,
            )

        texts: List[str] = [chunk.text for chunk in chunks]
        matrix = self._backend.encode(texts, batch_size=self._batch_size)
        normalised = self._normalise(matrix)

        return EmbeddedChunks(
            chunks=list(chunks),  # defensive copy preserves caller's list
            embeddings=normalised,
            model_name=self._model_name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        L2-normalise each row of a 2-D float32 matrix using epsilon for numerical stability.

        Args:
            matrix: 2-D NumPy array of shape (N, D).

        Returns:
            L2-normalised float32 array of the same shape.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # (N, 1)
        safe_norms = np.maximum(norms, 1e-12)
        return (matrix / safe_norms).astype(np.float32)



# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: List[KnowledgeChunk],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> "EmbeddedChunks":
    """
    Convenience entry point: embed a list of chunks with a fresh EmbeddingModel.

    Equivalent to::

        EmbeddingModel(model_name, batch_size).embed_chunks(chunks)

    When embedding multiple batches of chunks in the same process, prefer
    constructing a single EmbeddingModel instance and reusing it, to avoid
    loading the model repeatedly.

    Args:
        chunks:     List of KnowledgeChunk objects.
        model_name: HuggingFace model ID. Defaults to DEFAULT_MODEL_NAME.
        batch_size: Texts per encoding batch. Defaults to DEFAULT_BATCH_SIZE.

    Returns:
        EmbeddedChunks with float32 L2-normalised embeddings.
    """
    model = EmbeddingModel(model_name=model_name, batch_size=batch_size)
    return model.embed_chunks(chunks)
