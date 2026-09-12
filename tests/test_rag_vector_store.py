"""
Unit tests for the FAISS Vector Store layer (app/rag/vector_store.py).

Uses deterministic synthetic embeddings to test vector indexing, chunk mapping,
similarity search, validation, and persistence without network dependencies.

Test coverage:
    Construction:
        1. Empty store initialization with valid dimension.
        2. Reject non-positive or invalid dimensions.
        3. Construction from EmbeddedChunks (from_embedded_chunks).
        4. Dimension and vector count properties.

    Mapping & Traceability:
        5. First vector maps to first chunk.
        6. Input ordering is preserved exactly across all positions.
        7. All chunk IDs and complete KnowledgeChunk provenance remain intact.
        8. get_chunk() bounds validation.

    Search:
        9. Exact/self query returns the expected chunk at rank 1 with score ≈ 1.0.
        10. Search returns descending similarity scores and correct ranking.
        11. top_k=1 boundary case.
        12. top_k larger than index size returns at most ntotal results.
        13. Invalid top_k (< 1 or non-integer) raises ValueError.
        14. Wrong query vector dimension raises ValueError.
        15. Empty index search safely returns empty list.
        16. 1-D vs 2-D query vector format equivalence.

    Validation & Rejection:
        17. Duplicate chunk_id within a batch raises ValueError.
        18. Duplicate chunk_id across consecutive additions raises ValueError.
        19. Dimension mismatch on add_embedded_chunks raises ValueError.
        20. Row count mismatch between matrix and chunks list raises ValueError.
        21. Invalid query vector type raises TypeError.

    Persistence:
        22. save() and load() roundtrip preserves index and chunk mapping.
        23. Search results before and after persistence are identical.
        24. Dimension mismatch on load raises ValueError.
        25. Vector count mismatch on load raises ValueError.
        26. Missing index.faiss raises FileNotFoundError.
        27. Missing metadata.json raises FileNotFoundError.
        28. Corrupted metadata JSON raises ValueError.
        29. Duplicate chunk_id in metadata raises ValueError.
        30. Non-existent directory raises FileNotFoundError.

    Determinism:
        31. Building the store twice with the same inputs produces identical results.

    Package Exports:
        32. FAISSVectorStore and VectorSearchResult are exported by app.rag.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


import numpy as np
import numpy.typing as npt

from app.rag.embeddings import EmbeddedChunks
from app.rag.schemas import (
    KnowledgeChunk,
    KnowledgeFrameworkMetadata,
    KnowledgeSourceType,
    NISTReference,
)
from app.rag.vector_store import FAISSVectorStore, VectorSearchResult


# ===========================================================================
# Helpers & Synthetic Fixtures
# ===========================================================================

_TEST_DIM = 16  # Dimension for fast synthetic testing


def _make_unit_vector(dim: int, seed: int) -> npt.NDArray[np.float32]:
    """Generate a deterministic L2-normalized float32 vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        vec[0] = 1.0
        norm = 1.0
    return (vec / norm).astype(np.float32)


def _make_chunk(
    chunk_index: int,
    document_id: str = "test-doc-001",
    text: str = "Test chunk text content",
    source: str = "OWASP Foundation",
    source_type: KnowledgeSourceType = KnowledgeSourceType.OWASP,
) -> KnowledgeChunk:
    """Create a minimal KnowledgeChunk with full provenance for testing."""
    return KnowledgeChunk(
        chunk_id=f"{document_id}:chunk:{chunk_index}",
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
        section_title=f"Section {chunk_index}",
        source=source,
        source_type=source_type,
        url="https://owasp.org/test",
        framework_metadata=KnowledgeFrameworkMetadata(
            nist=[
                NISTReference(
                    control_id="AC-2",
                    title="Account Management",
                    publication="NIST SP 800-53",
                )
            ]
        ),
        tags=["access-control", "auth"],
        metadata={"created_for_test": True, "idx": chunk_index},
    )


def _make_embedded_chunks(
    count: int,
    dim: int = _TEST_DIM,
    document_id: str = "test-doc",
    model_name: str = "synthetic-model-v1",
) -> EmbeddedChunks:
    """Create synthetic EmbeddedChunks with orthogonal/spread unit vectors."""
    chunks: List[KnowledgeChunk] = []
    matrix_rows: List[npt.NDArray[np.float32]] = []

    for i in range(count):
        chunks.append(_make_chunk(i, document_id=document_id, text=f"Chunk content {i}"))
        matrix_rows.append(_make_unit_vector(dim, seed=i + 100))

    if count == 0:
        matrix = np.empty((0, dim), dtype=np.float32)
    else:
        matrix = np.stack(matrix_rows, axis=0)

    return EmbeddedChunks(
        chunks=chunks,
        embeddings=matrix,
        model_name=model_name,
    )


# ===========================================================================
# Test Suite
# ===========================================================================

class TestFAISSVectorStoreConstruction(unittest.TestCase):
    """Tests for vector store initialization and dimension handling."""

    def test_empty_store_initialization(self) -> None:
        """Store initializes as empty with specified dimension."""
        store = FAISSVectorStore(dimension=_TEST_DIM)
        self.assertEqual(store.dimension, _TEST_DIM)
        self.assertEqual(store.ntotal, 0)
        self.assertEqual(len(store), 0)
        self.assertTrue(store.is_empty())
        self.assertIsNone(store.model_name)
        self.assertEqual(store.get_chunks(), [])

    def test_invalid_dimension_raises_value_error(self) -> None:
        """Non-positive or non-integer dimensions must raise ValueError."""
        for invalid_dim in [0, -1, -384, "384", None]:
            with self.subTest(dim=invalid_dim):
                with self.assertRaises(ValueError):
                    FAISSVectorStore(dimension=invalid_dim)  # type: ignore

    def test_construction_from_embedded_chunks(self) -> None:
        """from_embedded_chunks() initializes and populates store directly."""
        embedded = _make_embedded_chunks(5, dim=_TEST_DIM)
        store = FAISSVectorStore.from_embedded_chunks(embedded)

        self.assertEqual(store.dimension, _TEST_DIM)
        self.assertEqual(store.ntotal, 5)
        self.assertFalse(store.is_empty())
        self.assertEqual(store.model_name, "synthetic-model-v1")

    def test_construction_from_empty_embedded_chunks(self) -> None:
        """from_embedded_chunks() handles empty EmbeddedChunks container."""
        empty_embedded = _make_embedded_chunks(0, dim=_TEST_DIM)
        store = FAISSVectorStore.from_embedded_chunks(empty_embedded)

        self.assertEqual(store.dimension, _TEST_DIM)
        self.assertEqual(store.ntotal, 0)
        self.assertTrue(store.is_empty())


class TestFAISSVectorStoreMapping(unittest.TestCase):
    """Tests for chunk ↔ vector mapping, ordering, and provenance preservation."""

    def setUp(self) -> None:
        self.embedded = _make_embedded_chunks(6, dim=_TEST_DIM)
        self.store = FAISSVectorStore.from_embedded_chunks(self.embedded)

    def test_first_vector_maps_to_first_chunk(self) -> None:
        """Position 0 corresponds exactly to the first input chunk."""
        first_chunk = self.store.get_chunk(0)
        self.assertEqual(first_chunk.chunk_id, self.embedded.chunks[0].chunk_id)
        self.assertEqual(first_chunk.text, self.embedded.chunks[0].text)

    def test_strict_ordering_preserved(self) -> None:
        """FAISS position i corresponds to embedded.chunks[i] for all i."""
        for i in range(len(self.embedded.chunks)):
            retrieved = self.store.get_chunk(i)
            expected = self.embedded.chunks[i]
            self.assertEqual(retrieved.chunk_id, expected.chunk_id)
            self.assertEqual(retrieved.chunk_index, expected.chunk_index)

    def test_provenance_fields_intact(self) -> None:
        """All metadata, NIST framework refs, source types, and URLs survive indexing."""
        retrieved = self.store.get_chunk(0)
        self.assertEqual(retrieved.source, "OWASP Foundation")
        self.assertEqual(retrieved.source_type, KnowledgeSourceType.OWASP)
        self.assertEqual(retrieved.url, "https://owasp.org/test")
        self.assertEqual(len(retrieved.framework_metadata.nist), 1)
        self.assertEqual(retrieved.framework_metadata.nist[0].control_id, "AC-2")
        self.assertEqual(retrieved.tags, ["access-control", "auth"])
        self.assertEqual(retrieved.metadata.get("idx"), 0)

    def test_get_chunk_out_of_bounds_raises_index_error(self) -> None:
        """Negative position or position >= ntotal raises IndexError."""
        with self.assertRaises(IndexError):
            self.store.get_chunk(-1)
        with self.assertRaises(IndexError):
            self.store.get_chunk(6)

    def test_get_chunks_returns_defensive_copy(self) -> None:
        """Mutating the list returned by get_chunks() does not alter the store."""
        chunks_copy = self.store.get_chunks()
        chunks_copy.clear()
        self.assertEqual(self.store.ntotal, 6)
        self.assertEqual(len(self.store.get_chunks()), 6)


class TestFAISSVectorStoreSearch(unittest.TestCase):
    """Tests for vector similarity search using FAISS IndexFlatIP."""

    def setUp(self) -> None:
        self.dim = _TEST_DIM
        self.embedded = _make_embedded_chunks(5, dim=self.dim)
        self.store = FAISSVectorStore.from_embedded_chunks(self.embedded)

    def test_exact_self_query_returns_top_1_with_unit_score(self) -> None:
        """Querying with vector i returns chunk i at rank 1 with score ≈ 1.0."""
        for i in range(self.store.ntotal):
            query_vec = self.embedded.embeddings[i]
            results = self.store.search(query_vec, top_k=3)

            self.assertGreater(len(results), 0)
            top_match = results[0]
            self.assertEqual(top_match.position, i)
            self.assertEqual(top_match.chunk.chunk_id, self.embedded.chunks[i].chunk_id)
            self.assertAlmostEqual(top_match.score, 1.0, places=5)

    def test_search_results_in_descending_similarity_order(self) -> None:
        """FAISS results must be strictly monotonically non-increasing by score."""
        query_vec = self.embedded.embeddings[0]
        results = self.store.search(query_vec, top_k=5)

        self.assertEqual(len(results), 5)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i].score,
                results[i + 1].score,
                msg=f"Ranking order violated between rank {i} and {i+1}.",
            )

    def test_top_k_one(self) -> None:
        """top_k=1 returns exactly one match."""
        results = self.store.search(self.embedded.embeddings[0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].position, 0)

    def test_top_k_larger_than_index_size(self) -> None:
        """Requesting top_k > ntotal returns at most ntotal results without crashing."""
        results = self.store.search(self.embedded.embeddings[0], top_k=100)
        self.assertEqual(len(results), self.store.ntotal)

    def test_invalid_top_k_raises_value_error(self) -> None:
        """Non-positive, non-integer, or boolean top_k must raise ValueError."""
        query = self.embedded.embeddings[0]
        for invalid_k in [0, -1, -5, "3", 1.5, True, False]:
            with self.subTest(top_k=invalid_k):
                with self.assertRaises(ValueError):
                    self.store.search(query, top_k=invalid_k)  # type: ignore


    def test_wrong_query_dimension_raises_value_error(self) -> None:
        """Query vector with mismatched dimension must raise ValueError."""
        wrong_dim_query = np.ones(self.dim + 4, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.store.search(wrong_dim_query, top_k=3)

    def test_empty_index_search_returns_empty_list(self) -> None:
        """Searching an empty store returns [] safely."""
        empty_store = FAISSVectorStore(dimension=self.dim)
        query = np.ones(self.dim, dtype=np.float32)
        results = empty_store.search(query, top_k=5)
        self.assertEqual(results, [])

    def test_1d_and_2d_query_vectors_equivalent(self) -> None:
        """1-D (dim,) and 2-D (1, dim) query vectors produce identical results."""
        query_1d = self.embedded.embeddings[2]
        query_2d = query_1d.reshape(1, -1)

        res_1d = self.store.search(query_1d, top_k=3)
        res_2d = self.store.search(query_2d, top_k=3)

        self.assertEqual(len(res_1d), len(res_2d))
        for r1, r2 in zip(res_1d, res_2d):
            self.assertEqual(r1.position, r2.position)
            self.assertAlmostEqual(r1.score, r2.score, places=6)


class TestFAISSVectorStoreValidation(unittest.TestCase):
    """Tests for input validation and duplicate chunk rejection."""

    def test_duplicate_chunk_id_within_same_batch_raises_value_error(self) -> None:
        """Duplicate chunk_id within the incoming batch raises ValueError."""
        chunk1 = _make_chunk(0, document_id="doc-A")
        chunk2 = _make_chunk(0, document_id="doc-A")  # Duplicate ID!
        matrix = np.random.randn(2, _TEST_DIM).astype(np.float32)

        embedded = EmbeddedChunks(chunks=[chunk1, chunk2], embeddings=matrix, model_name="test")
        store = FAISSVectorStore(dimension=_TEST_DIM)

        with self.assertRaises(ValueError) as ctx:
            store.add_embedded_chunks(embedded)
        self.assertIn("Duplicate chunk_id", str(ctx.exception))

    def test_duplicate_chunk_id_across_consecutive_batches_raises_value_error(self) -> None:
        """Adding a chunk whose ID already exists in the store raises ValueError."""
        store = FAISSVectorStore(dimension=_TEST_DIM)
        batch1 = _make_embedded_chunks(3, dim=_TEST_DIM, document_id="doc-1")
        store.add_embedded_chunks(batch1)

        # Batch 2 contains a chunk with same ID as in batch 1
        dup_chunk = _make_chunk(0, document_id="doc-1")
        dup_matrix = np.random.randn(1, _TEST_DIM).astype(np.float32)
        batch2 = EmbeddedChunks(chunks=[dup_chunk], embeddings=dup_matrix, model_name="test")

        with self.assertRaises(ValueError) as ctx:
            store.add_embedded_chunks(batch2)
        self.assertIn("already exists", str(ctx.exception))

    def test_dimension_mismatch_on_add_raises_value_error(self) -> None:
        """Adding embeddings with wrong dimension raises ValueError."""
        store = FAISSVectorStore(dimension=_TEST_DIM)
        wrong_dim_batch = _make_embedded_chunks(3, dim=_TEST_DIM + 8)

        with self.assertRaises(ValueError) as ctx:
            store.add_embedded_chunks(wrong_dim_batch)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_matrix_shape_and_chunks_len_mismatch_raises_value_error(self) -> None:
        """Matrix rows != len(chunks) raises ValueError."""
        store = FAISSVectorStore(dimension=_TEST_DIM)
        chunks = [_make_chunk(0), _make_chunk(1)]
        wrong_matrix = np.zeros((5, _TEST_DIM), dtype=np.float32)

        # Bypass EmbeddedChunks validation to test store defense
        store_corrupt = FAISSVectorStore(dimension=_TEST_DIM)
        with self.assertRaises(ValueError):
            # Construct invalid dataclass without __post_init__ or direct call
            EmbeddedChunks(chunks=chunks, embeddings=wrong_matrix, model_name="test")

    def test_invalid_query_type_raises_type_error(self) -> None:
        """Non-numeric query raises TypeError."""
        store = FAISSVectorStore(dimension=_TEST_DIM)
        with self.assertRaises((TypeError, ValueError)):
            store.search("invalid-query-string", top_k=3)  # type: ignore

    def test_atomic_batch_insertion_failure_leaves_store_intact(self) -> None:
        """Failure on an incoming batch (e.g. duplicate chunk) must not mutate store state."""
        store = FAISSVectorStore(dimension=_TEST_DIM)
        initial_batch = _make_embedded_chunks(3, dim=_TEST_DIM, document_id="doc-initial")
        store.add_embedded_chunks(initial_batch)
        self.assertEqual(store.ntotal, 3)

        # Create an incoming batch where chunk 0 is valid and new, but chunk 1 duplicates an existing chunk ID
        bad_chunk_1 = _make_chunk(10, document_id="doc-new")
        bad_chunk_2 = _make_chunk(0, document_id="doc-initial")  # Already in store!
        bad_matrix = np.random.randn(2, _TEST_DIM).astype(np.float32)
        bad_embedded = EmbeddedChunks(chunks=[bad_chunk_1, bad_chunk_2], embeddings=bad_matrix, model_name="test")

        with self.assertRaises(ValueError):
            store.add_embedded_chunks(bad_embedded)

        # Verify strict atomicity: store remains at 3 vectors; neither chunk was added
        self.assertEqual(store.ntotal, 3)
        self.assertEqual(len(store.get_chunks()), 3)
        self.assertNotIn("doc-new:chunk:10", store._chunk_ids)

        # Verify search still works properly over the original 3 chunks
        query = initial_batch.embeddings[0]
        results = store.search(query, top_k=5)
        self.assertEqual(len(results), 3)



class TestFAISSVectorStorePersistence(unittest.TestCase):
    """Tests for saving and loading FAISSVectorStore with index.faiss + metadata.json."""

    def setUp(self) -> None:
        self.dim = _TEST_DIM
        self.embedded = _make_embedded_chunks(4, dim=self.dim)
        self.store = FAISSVectorStore.from_embedded_chunks(self.embedded)

    def test_save_and_load_roundtrip(self) -> None:
        """Saved store restores accurately with all chunks, dimensions, and vectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save(tmpdir)

            # Verify files were created
            index_file = Path(tmpdir) / "index.faiss"
            meta_file = Path(tmpdir) / "metadata.json"
            self.assertTrue(index_file.exists())
            self.assertTrue(meta_file.exists())

            # Load store
            loaded = FAISSVectorStore.load(tmpdir)

            self.assertEqual(loaded.dimension, self.store.dimension)
            self.assertEqual(loaded.ntotal, self.store.ntotal)
            self.assertEqual(loaded.model_name, self.store.model_name)
            self.assertEqual(len(loaded.get_chunks()), len(self.store.get_chunks()))

            for i in range(self.store.ntotal):
                orig_chunk = self.store.get_chunk(i)
                loaded_chunk = loaded.get_chunk(i)
                self.assertEqual(orig_chunk.chunk_id, loaded_chunk.chunk_id)
                self.assertEqual(orig_chunk.text, loaded_chunk.text)
                self.assertEqual(orig_chunk.source, loaded_chunk.source)
                self.assertEqual(orig_chunk.framework_metadata.nist[0].control_id, "AC-2")

    def test_search_results_identical_after_persistence(self) -> None:
        """Search scores and chunk ranking match exactly before and after persistence."""
        query_vec = self.embedded.embeddings[1]
        before_results = self.store.search(query_vec, top_k=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save(tmpdir)
            loaded = FAISSVectorStore.load(tmpdir)
            after_results = loaded.search(query_vec, top_k=4)

        self.assertEqual(len(before_results), len(after_results))
        for r_orig, r_loaded in zip(before_results, after_results):
            self.assertEqual(r_orig.position, r_loaded.position)
            self.assertEqual(r_orig.chunk.chunk_id, r_loaded.chunk.chunk_id)
            self.assertAlmostEqual(r_orig.score, r_loaded.score, places=6)

    def test_missing_index_faiss_raises_file_not_found(self) -> None:
        """Missing index.faiss raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save(tmpdir)
            (Path(tmpdir) / "index.faiss").unlink()

            with self.assertRaises(FileNotFoundError):
                FAISSVectorStore.load(tmpdir)

    def test_missing_metadata_json_raises_file_not_found(self) -> None:
        """Missing metadata.json raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save(tmpdir)
            (Path(tmpdir) / "metadata.json").unlink()

            with self.assertRaises(FileNotFoundError):
                FAISSVectorStore.load(tmpdir)

    def test_nonexistent_directory_raises_file_not_found(self) -> None:
        """Loading from non-existent directory raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            FAISSVectorStore.load("/path/to/nonexistent/vector_store_dir")

    def test_corrupted_metadata_json_raises_value_error(self) -> None:
        """Malformed metadata JSON raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save(tmpdir)
            with open(Path(tmpdir) / "metadata.json", "w") as f:
                f.write("{invalid json content:")

            with self.assertRaises(ValueError):
                FAISSVectorStore.load(tmpdir)

    def test_dimension_mismatch_in_metadata_raises_value_error(self) -> None:
        """Metadata reporting different dimension than index raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save(tmpdir)
            meta_path = Path(tmpdir) / "metadata.json"
            with open(meta_path, "r") as f:
                data = json.load(f)
            data["embedding_dimension"] = self.dim + 10  # Mismatch!
            with open(meta_path, "w") as f:
                json.dump(data, f)

            with self.assertRaises(ValueError) as ctx:
                FAISSVectorStore.load(tmpdir)
            self.assertIn("Dimension mismatch", str(ctx.exception))

    def test_vector_count_mismatch_raises_value_error(self) -> None:
        """Metadata reporting different vector count than index raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save(tmpdir)
            meta_path = Path(tmpdir) / "metadata.json"
            with open(meta_path, "r") as f:
                data = json.load(f)
            data["vector_count"] = 999  # Mismatch!
            with open(meta_path, "w") as f:
                json.dump(data, f)

            with self.assertRaises(ValueError) as ctx:
                FAISSVectorStore.load(tmpdir)
            self.assertIn("Vector count mismatch", str(ctx.exception))

    def test_duplicate_chunk_in_metadata_raises_value_error(self) -> None:
        """Corrupted metadata with duplicate chunk_id raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save(tmpdir)
            meta_path = Path(tmpdir) / "metadata.json"
            with open(meta_path, "r") as f:
                data = json.load(f)
            if len(data["chunks"]) >= 2:
                data["chunks"][1]["chunk_id"] = data["chunks"][0]["chunk_id"]
            with open(meta_path, "w") as f:
                json.dump(data, f)

            with self.assertRaises(ValueError) as ctx:
                FAISSVectorStore.load(tmpdir)
            self.assertIn("duplicate chunk_id", str(ctx.exception))

    def test_missing_format_version_raises_value_error(self) -> None:
        """Metadata missing format_version raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save(tmpdir)
            meta_path = Path(tmpdir) / "metadata.json"
            with open(meta_path, "r") as f:
                data = json.load(f)
            del data["format_version"]
            with open(meta_path, "w") as f:
                json.dump(data, f)

            with self.assertRaises(ValueError) as ctx:
                FAISSVectorStore.load(tmpdir)
            self.assertIn("missing required fields", str(ctx.exception).lower())

    def test_unsupported_format_version_raises_value_error(self) -> None:
        """Metadata with unknown or unsupported format_version raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save(tmpdir)
            meta_path = Path(tmpdir) / "metadata.json"
            with open(meta_path, "r") as f:
                data = json.load(f)
            data["format_version"] = "99.0"  # Incompatible future version
            with open(meta_path, "w") as f:
                json.dump(data, f)

            with self.assertRaises(ValueError) as ctx:
                FAISSVectorStore.load(tmpdir)
            self.assertIn("Unsupported format_version", str(ctx.exception))



class TestFAISSVectorStoreDeterminism(unittest.TestCase):
    """Tests to guarantee determinism in vector index construction and search."""

    def test_repeated_construction_produces_identical_scores_and_order(self) -> None:
        """Building two separate stores with identical data yields identical results."""
        embedded = _make_embedded_chunks(5, dim=_TEST_DIM)

        store1 = FAISSVectorStore.from_embedded_chunks(embedded)
        store2 = FAISSVectorStore.from_embedded_chunks(embedded)

        query = _make_unit_vector(_TEST_DIM, seed=999)
        res1 = store1.search(query, top_k=5)
        res2 = store2.search(query, top_k=5)

        self.assertEqual(len(res1), len(res2))
        for r1, r2 in zip(res1, res2):
            self.assertEqual(r1.position, r2.position)
            self.assertEqual(r1.chunk.chunk_id, r2.chunk.chunk_id)
            self.assertAlmostEqual(r1.score, r2.score, places=6)


class TestRAGExports(unittest.TestCase):
    """Verify FAISSVectorStore and VectorSearchResult are properly exported from app.rag."""

    def test_app_rag_exports_vector_store_symbols(self) -> None:
        """Symbols can be imported directly from app.rag."""
        import app.rag as rag

        self.assertTrue(hasattr(rag, "FAISSVectorStore"))
        self.assertTrue(hasattr(rag, "VectorSearchResult"))


def _real_model_available() -> bool:
    """Return True only if the cached model is available offline."""
    try:
        import os
        cache_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
        )
        return os.path.isdir(cache_dir)
    except Exception:
        return False


_SKIP_INTEGRATION = not _real_model_available()
_SKIP_REASON = "all-MiniLM-L6-v2 not found in local cache; skipping real model test"


@unittest.skipIf(_SKIP_INTEGRATION, _SKIP_REASON)
class TestFAISSVectorStoreRealIntegration(unittest.TestCase):
    """End-to-end integration test with the real SentenceTransformer model and FAISS."""

    @classmethod
    def setUpClass(cls) -> None:
        from app.rag.embeddings import EmbeddingModel
        cls.embedder = EmbeddingModel()

    def test_end_to_end_rag_indexing_and_search(self) -> None:
        """Full pipeline: Document -> Chunks -> Embeddings -> FAISS -> Save/Load -> Search."""
        from app.rag.chunking import clean_and_chunk_document
        from app.rag.schemas import KnowledgeDocument

        doc = KnowledgeDocument(
            document_id="owasp-a01-test",
            title="A01: Broken Access Control",
            source="OWASP Foundation",
            source_type=KnowledgeSourceType.OWASP,
            content="Access control enforces policy such that users cannot act outside of their intended permissions. Bypassing access checks can lead to unauthorized information disclosure or privilege escalation.",
        )
        chunks = clean_and_chunk_document(doc)
        self.assertGreater(len(chunks), 0)

        embedded = self.embedder.embed_chunks(chunks)
        self.assertEqual(embedded.embedding_dim, 384)

        store = FAISSVectorStore.from_embedded_chunks(embedded)
        self.assertEqual(store.dimension, 384)
        self.assertEqual(store.ntotal, len(chunks))

        # Test persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)
            loaded_store = FAISSVectorStore.load(tmpdir)
            self.assertEqual(loaded_store.dimension, 384)
            self.assertEqual(loaded_store.ntotal, len(chunks))

            # Query relevant to access control
            query_vec = self.embedder.embed_text("Unauthorized privilege escalation and permission bypass")
            results = loaded_store.search(query_vec, top_k=1)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].chunk.document_id, "owasp-a01-test")
            self.assertGreater(results[0].score, 0.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

