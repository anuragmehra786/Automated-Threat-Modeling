"""
Cybersecurity Knowledge Base — Vector Store Layer (FAISS).

Indexes dense embedding vectors using FAISS IndexFlatIP (exact inner-product
search over L2-normalized embeddings) and maintains deterministic,
provenance-preserving mappings back to the source KnowledgeChunk objects.

Pipeline position:
    KnowledgeChunk      (from chunking.py)
        ↓
    EmbeddingModel      (from embeddings.py)
        ↓
    EmbeddedChunks      (normalized float32 vectors)
        ↓
    FAISSVectorStore    (this module)
        ↓
    NEXT STEP: Semantic Retrieval / Hybrid Search / Context Assembly

Design Principles:
    1. Single Responsibility: Vector indexing, exact search, and chunk mapping.
       No retrieval ranking, no BM25, no re-ranking, no LLM reasoning.
    2. Exact Cosine Search: Uses faiss.IndexFlatIP. Because Step 4 guarantees
       L2-normalized vectors, inner product is mathematically identical to
       cosine similarity.
    3. Strict Traceability: Maintains a deterministic 1-to-1 mapping:
       FAISS position i  <->  KnowledgeChunk i
       Every retrieved result preserves the complete parent KnowledgeChunk.
    4. Contract Integrity: The store expects L2-normalized float32 vectors.
       It does NOT silently re-normalize or truncate input vectors.
    5. Duplicate Rejection: Inserting duplicate chunk_id values within the
       same store raises a descriptive ValueError.
    6. Persistence: Saves and loads both the binary FAISS index (index.faiss)
       and the structured Pydantic chunk metadata (metadata.json) together,
       validating dimensional and count consistency upon reload.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

# Prevent macOS duplicate OpenMP runtime conflict when both PyTorch and FAISS are loaded
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# pyrefly: ignore [missing-import]
import faiss
import numpy as np
import numpy.typing as npt


from app.rag.embeddings import EmbeddedChunks
from app.rag.schemas import KnowledgeChunk

logger = logging.getLogger(__name__)

#: Current persistence format version.
CURRENT_FORMAT_VERSION: str = "1.0"

#: Supported format versions for loading.
SUPPORTED_FORMAT_VERSIONS: Set[str] = {"1.0"}



# ---------------------------------------------------------------------------
# Search Result Container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VectorSearchResult:
    """
    Result of a vector similarity search in FAISSVectorStore.

    Attributes:
        chunk:    The matched KnowledgeChunk preserving complete provenance.
        score:    The inner-product similarity score (cosine similarity for
                  L2-normalized vectors). Higher is more similar.
        position: The integer position index in the FAISS index.
    """

    chunk: KnowledgeChunk
    score: float
    position: int


# ---------------------------------------------------------------------------
# FAISS Vector Store
# ---------------------------------------------------------------------------

class FAISSVectorStore:
    """
    In-memory vector store backed by a FAISS IndexFlatIP inner-product index.

    Maintains a deterministic mapping between integer FAISS vector positions
    and rich, validated KnowledgeChunk objects.

    Contract:
        - Vectors supplied to this store are expected to be L2-normalized
          float32 vectors (as produced by app.rag.embeddings.EmbeddingModel).
        - The store does not silently alter or re-normalize vectors.
        - Dimensions must match the store's configured dimension exactly.

    Args:
        dimension: Dimensionality of the embedding vectors (e.g. 384).
                   Must be a positive integer.

    Raises:
        ValueError: If dimension <= 0.
    """

    def __init__(self, dimension: int) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(
                f"FAISSVectorStore dimension must be a positive integer, got {dimension!r}."
            )

        self._dimension: int = dimension
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self._chunks: List[KnowledgeChunk] = []
        self._chunk_ids: Set[str] = set()
        self._model_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Embedding dimension of the underlying FAISS index."""
        return int(self._index.d)

    @property
    def ntotal(self) -> int:
        """Total number of vectors indexed in the store."""
        return int(self._index.ntotal)

    @property
    def model_name(self) -> Optional[str]:
        """Identifier of the embedding model used, if recorded."""
        return self._model_name

    def is_empty(self) -> bool:
        """Return True if the vector store contains no vectors, False otherwise."""
        return self.ntotal == 0

    def __len__(self) -> int:
        """Return the number of vectors in the store."""
        return self.ntotal

    def get_chunk(self, position: int) -> KnowledgeChunk:
        """
        Retrieve a KnowledgeChunk by its integer position in the index.

        Args:
            position: 0-based integer index.

        Returns:
            The KnowledgeChunk at the specified position.

        Raises:
            IndexError: If position is out of bounds.
        """
        if not (0 <= position < len(self._chunks)):
            raise IndexError(
                f"Position {position} out of bounds for vector store with {len(self._chunks)} chunks."
            )
        return self._chunks[position]

    def get_chunks(self) -> List[KnowledgeChunk]:
        """Return a defensive shallow copy of all indexed KnowledgeChunk objects in order."""
        return list(self._chunks)

    # ------------------------------------------------------------------
    # Ingestion / Population
    # ------------------------------------------------------------------

    @classmethod
    def from_embedded_chunks(cls, embedded: EmbeddedChunks) -> "FAISSVectorStore":
        """
        Construct and populate a FAISSVectorStore directly from an EmbeddedChunks instance.

        Args:
            embedded: An EmbeddedChunks container containing chunks and float32 matrix.

        Returns:
            Populated FAISSVectorStore instance.

        Raises:
            ValueError: If embedded has an invalid or non-positive dimension.
        """
        dim = embedded.embedding_dim
        if dim <= 0:
            raise ValueError(
                f"Cannot create FAISSVectorStore from EmbeddedChunks with invalid dimension {dim}."
            )

        store = cls(dimension=dim)
        store.add_embedded_chunks(embedded)
        return store

    def add_embedded_chunks(self, embedded: EmbeddedChunks) -> None:
        """
        Add a batch of embedded chunks to the FAISS index and internal mapping.

        Input order is preserved deterministically:
            FAISS index (old_ntotal + i) <-> embedded.chunks[i]

        Args:
            embedded: An EmbeddedChunks container.

        Raises:
            ValueError: If embedding dimension does not match store dimension,
                        if matrix shape does not match chunk count, or
                        if any duplicate chunk_id is detected.
        """
        if embedded.embedding_dim != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: store requires {self.dimension}, "
                f"but received {embedded.embedding_dim}."
            )

        matrix = embedded.embeddings
        chunks = embedded.chunks

        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Expected 2-D embedding matrix of shape (N, {self.dimension}), "
                f"got shape {matrix.shape}."
            )

        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding matrix row count ({matrix.shape[0]}) does not match "
                f"chunks list length ({len(chunks)})."
            )

        # Handle empty input safely
        if len(chunks) == 0:
            if self._model_name is None and embedded.model_name:
                self._model_name = embedded.model_name
            return

        # Check for duplicates within the incoming batch
        incoming_ids: Set[str] = set()
        for chunk in chunks:
            cid = chunk.chunk_id
            if cid in incoming_ids:
                raise ValueError(
                    f"Duplicate chunk_id '{cid}' detected within the incoming batch."
                )
            if cid in self._chunk_ids:
                raise ValueError(
                    f"Duplicate chunk_id '{cid}' detected: chunk already exists in vector store."
                )
            incoming_ids.add(cid)

        # Ensure float32 contiguous numpy array for FAISS
        vectors = np.ascontiguousarray(matrix, dtype=np.float32)

        # Add to FAISS index
        self._index.add(vectors)

        # Append chunks and record IDs in exact alignment
        for chunk in chunks:
            self._chunks.append(chunk)
            self._chunk_ids.add(chunk.chunk_id)

        # Record model name from first non-empty ingestion if not already set
        if self._model_name is None and embedded.model_name:
            self._model_name = embedded.model_name

        logger.debug(
            "Added %d chunks to FAISSVectorStore. Total indexed: %d.",
            len(chunks),
            self.ntotal,
        )

    # ------------------------------------------------------------------
    # Search API
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: Union[npt.NDArray[np.float32], Sequence[float]],
        top_k: int = 5,
    ) -> List[VectorSearchResult]:
        """
        Perform exact vector similarity search using FAISS IndexFlatIP.

        Computes the inner product between query_vector and all indexed vectors,
        returning top_k matches sorted in descending order of similarity.

        Args:
            query_vector: 1-D array of shape (dimension,) or 2-D array of shape (1, dimension).
                          Must be float32 compatible and match the store's dimension.
            top_k:        Maximum number of matches to return (must be >= 1).

        Returns:
            List of VectorSearchResult ordered by similarity score (descending).
            Returns an empty list if the store is empty.
            Never returns more results than the number of vectors in the store.

        Raises:
            ValueError: If top_k < 1 or query_vector dimension does not match.
            TypeError:  If query_vector cannot be converted to a numeric array.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer >= 1, got {top_k!r}.")


        # Convert query to numpy array
        try:
            query_arr = np.asarray(query_vector, dtype=np.float32)
        except Exception as exc:
            raise TypeError(f"Could not convert query_vector to float32 NumPy array: {exc}") from exc

        # Validate shape
        if query_arr.ndim == 1:
            if query_arr.shape[0] != self.dimension:
                raise ValueError(
                    f"Query vector dimension mismatch: expected {self.dimension}, got {query_arr.shape[0]}."
                )
            query_arr = query_arr.reshape(1, -1)
        elif query_arr.ndim == 2:
            if query_arr.shape[0] != 1 or query_arr.shape[1] != self.dimension:
                raise ValueError(
                    f"Query vector shape mismatch: expected (1, {self.dimension}), got {query_arr.shape}."
                )
        else:
            raise ValueError(
                f"Query vector must be 1-D or 2-D, got {query_arr.ndim}-D array with shape {query_arr.shape}."
            )

        # Empty store returns empty results safely without error
        if self.is_empty():
            return []

        # Ensure C-contiguous
        query_arr = np.ascontiguousarray(query_arr, dtype=np.float32)


        # Query at most min(top_k, self.ntotal) to not ask FAISS for more than exists
        k = min(top_k, self.ntotal)
        scores, indices = self._index.search(query_arr, k)

        results: List[VectorSearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # Sentinel returned by FAISS if insufficient neighbors found
                continue
            idx_int = int(idx)
            chunk = self._chunks[idx_int]
            results.append(
                VectorSearchResult(
                    chunk=chunk,
                    score=float(score),
                    position=idx_int,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Union[str, Path]) -> None:
        """
        Persist the FAISS vector index and chunk metadata to a local directory.

        Output files:
            <directory>/index.faiss    - Binary FAISS IndexFlatIP file
            <directory>/metadata.json  - Structured JSON containing dimension,
                                         counts, model info, and serialized KnowledgeChunks.

        Args:
            directory: Path to target directory (will be created if it does not exist).
        """
        dir_path = Path(directory).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        index_file = dir_path / "index.faiss"
        metadata_file = dir_path / "metadata.json"

        # 1. Save binary FAISS index
        faiss.write_index(self._index, str(index_file))

        # 2. Serialize chunk metadata via Pydantic model_dump
        chunks_data = [chunk.model_dump(mode="json") for chunk in self._chunks]

        metadata: Dict[str, Any] = {
            "format_version": CURRENT_FORMAT_VERSION,
            "embedding_dimension": self.dimension,
            "vector_count": self.ntotal,
            "model_name": self._model_name,
            "chunks": chunks_data,
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            "Successfully saved FAISSVectorStore (%d vectors, dim %d) to %s.",
            self.ntotal,
            self.dimension,
            dir_path,
        )

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "FAISSVectorStore":
        """
        Load a persisted FAISSVectorStore from a local directory.

        Validates that:
            - Both index.faiss and metadata.json exist.
            - Metadata is valid JSON and structurally sound.
            - format_version is supported.
            - FAISS index dimension matches metadata dimension.
            - FAISS index ntotal matches metadata vector count.
            - Number of deserialized chunks matches vector count.
            - All chunks deserialize into valid KnowledgeChunk objects.
            - No duplicate chunk_ids exist.

        Args:
            directory: Directory containing index.faiss and metadata.json.

        Returns:
            Fully restored FAISSVectorStore instance.

        Raises:
            FileNotFoundError: If the directory or required files do not exist.
            ValueError: If index/metadata are corrupted, mismatched, or invalid.
        """
        dir_path = Path(directory).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Vector store directory not found: {dir_path}")

        index_file = dir_path / "index.faiss"
        metadata_file = dir_path / "metadata.json"

        if not index_file.exists():
            raise FileNotFoundError(
                f"Missing FAISS index file: {index_file}. Directory is not a valid vector store."
            )
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Missing metadata file: {metadata_file}. Directory is not a valid vector store."
            )

        # 1. Load metadata JSON
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupted metadata JSON in '{metadata_file}': {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Failed to read metadata file '{metadata_file}': {exc}") from exc

        if not isinstance(metadata, dict):
            raise ValueError(f"Metadata file '{metadata_file}' must contain a JSON object.")

        required_keys = {"format_version", "embedding_dimension", "vector_count", "chunks"}
        missing_keys = required_keys - set(metadata.keys())
        if missing_keys:
            raise ValueError(
                f"Metadata file '{metadata_file}' is missing required fields: {sorted(missing_keys)}."
            )

        format_ver = metadata["format_version"]
        if not isinstance(format_ver, str) or format_ver not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"Unsupported format_version '{format_ver}' in metadata. "
                f"Supported versions: {sorted(SUPPORTED_FORMAT_VERSIONS)}."
            )

        meta_dim = metadata["embedding_dimension"]
        meta_count = metadata["vector_count"]
        chunks_raw = metadata["chunks"]


        if not isinstance(meta_dim, int) or meta_dim <= 0:
            raise ValueError(f"Invalid embedding_dimension in metadata: {meta_dim!r}.")
        if not isinstance(meta_count, int) or meta_count < 0:
            raise ValueError(f"Invalid vector_count in metadata: {meta_count!r}.")
        if not isinstance(chunks_raw, list):
            raise ValueError(f"Invalid chunks list in metadata, expected list, got {type(chunks_raw).__name__}.")

        # 2. Load FAISS index
        try:
            index = faiss.read_index(str(index_file))
        except Exception as exc:
            raise ValueError(f"Failed to read FAISS index file '{index_file}': {exc}") from exc

        # 3. Consistency checks
        if index.d != meta_dim:
            raise ValueError(
                f"Dimension mismatch between FAISS index ({index.d}) and metadata ({meta_dim})."
            )
        if index.ntotal != meta_count:
            raise ValueError(
                f"Vector count mismatch between FAISS index ({index.ntotal}) and metadata ({meta_count})."
            )
        if len(chunks_raw) != index.ntotal:
            raise ValueError(
                f"Chunks count in metadata ({len(chunks_raw)}) does not match FAISS vector count ({index.ntotal})."
            )

        # 4. Deserialise and validate KnowledgeChunks
        chunks: List[KnowledgeChunk] = []
        chunk_ids: Set[str] = set()
        for i, item in enumerate(chunks_raw):
            try:
                chunk = KnowledgeChunk.model_validate(item)
            except Exception as exc:
                raise ValueError(f"Failed to validate KnowledgeChunk at index {i}: {exc}") from exc

            if chunk.chunk_id in chunk_ids:
                raise ValueError(
                    f"Corrupted metadata: duplicate chunk_id '{chunk.chunk_id}' found at index {i}."
                )
            chunk_ids.add(chunk.chunk_id)
            chunks.append(chunk)

        # 5. Instantiate and populate store
        store = cls(dimension=meta_dim)
        store._index = index
        store._chunks = chunks
        store._chunk_ids = chunk_ids
        store._model_name = metadata.get("model_name")

        logger.info(
            "Successfully loaded FAISSVectorStore (%d vectors, dim %d) from %s.",
            store.ntotal,
            store.dimension,
            dir_path,
        )
        return store
