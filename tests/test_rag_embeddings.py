"""
Unit and integration tests for the RAG Embedding Layer (app/rag/embeddings.py).

Test strategy
-------------
* Unit tests (class TestEmbeddingModelUnit): use a lightweight deterministic
  mock backend — no network access, no model download, fast execution.
* Integration tests (class TestEmbeddingModelIntegration): use the real
  SentenceTransformerBackend with the locally-cached all-MiniLM-L6-v2 model.
  These are skipped automatically when the model is not available offline or
  when the sentence-transformers library is not installed.

Tests cover:
    1.  EmbeddingModel can be initialised with a mock backend.
    2.  embed_text() returns a vector of the expected dimension.
    3.  Vector dtype is float32.
    4.  embed_chunks() returns one vector per input chunk.
    5.  Input ordering is preserved (chunks[i] <-> embeddings[i]).
    6.  Empty chunk list returns shape (0, embedding_dim).
    7.  embed_text("") raises ValueError.
    8.  embed_text("   ") raises ValueError (whitespace-only).
    9.  Multiple calls reuse the same backend instance (no reload).
    10. Embeddings are L2-normalised (unit vectors).
    11. Same input text produces an equivalent embedding across repeated calls.
    12. Semantically related text pair scores higher than an unrelated pair.
    13. EmbeddedChunks invariant: len(chunks) == embeddings.shape[0].
    14. EmbeddingModel.embedding_dim matches actual vector length.
    15. Integration: real model initialises with correct dimension (384).
    16. Integration: real embed_text returns float32 unit vector of dim 384.
    17. Integration: real embed_chunks preserves order and shape.
    18. Integration: semantic similarity test with real model.
"""

import unittest
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt

from app.rag.embeddings import (
    DEFAULT_MODEL_NAME,
    EmbeddedChunks,
    EmbeddingModel,
    SentenceTransformerBackend,
    embed_chunks,
)
from app.rag.schemas import KnowledgeChunk, KnowledgeSourceType


# ===========================================================================
# Helpers / Fixtures
# ===========================================================================

_MOCK_DIM = 8  # Tiny dimension for fast unit tests


class MockBackend:
    """
    Deterministic mock embedding backend for unit tests.

    encode() returns a reproducible array whose rows are derived from the
    hash of each input text, so the same text always produces the same vector.
    The returned vectors are NOT pre-normalised; normalisation is the
    responsibility of EmbeddingModel._normalise().
    """

    def __init__(self, dim: int = _MOCK_DIM) -> None:
        self._dim = dim
        self.encode_call_count: int = 0

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> npt.NDArray[np.float32]:
        self.encode_call_count += 1
        rows = []
        for text in texts:
            # Deterministic vector from hash: same text -> same numbers.
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            row = rng.random(self._dim).astype(np.float32) + 0.1  # avoid zero-norm
            rows.append(row)
        return np.stack(rows, axis=0)


def _make_chunk(
    text: str,
    chunk_index: int = 0,
    document_id: str = "test-doc",
) -> KnowledgeChunk:
    """Minimal KnowledgeChunk factory for testing."""
    return KnowledgeChunk(
        chunk_id=f"{document_id}:chunk:{chunk_index}",
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
        source="OWASP Foundation",
        source_type=KnowledgeSourceType.OWASP,
    )


def _make_chunks(texts: List[str], document_id: str = "test-doc") -> List[KnowledgeChunk]:
    """Build an ordered list of KnowledgeChunks from a list of texts."""
    return [_make_chunk(text, i, document_id) for i, text in enumerate(texts)]


def _cosine_sim(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ===========================================================================
# Unit Tests — mock backend, no network
# ===========================================================================

class TestEmbeddingModelUnit(unittest.TestCase):
    """
    Fast unit tests using a deterministic mock backend.

    No sentence-transformers, no torch, no network required.
    """

    def setUp(self) -> None:
        self.mock_backend = MockBackend(dim=_MOCK_DIM)
        self.embedder = EmbeddingModel(
            model_name="mock-model",
            backend=self.mock_backend,
        )

    # -----------------------------------------------------------------------
    # Test 1 — Model can be initialised
    # -----------------------------------------------------------------------
    def test_model_initialises_with_mock_backend(self) -> None:
        """EmbeddingModel initialises without error when given a mock backend."""
        embedder = EmbeddingModel(model_name="mock-model", backend=MockBackend())
        self.assertIsNotNone(embedder)
        self.assertEqual(embedder.model_name, "mock-model")

    # -----------------------------------------------------------------------
    # Test 2 — Single text produces a vector
    # -----------------------------------------------------------------------
    def test_embed_text_returns_vector(self) -> None:
        """embed_text() returns a 1-D numpy array."""
        vector = self.embedder.embed_text("SQL injection bypass authentication.")
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.ndim, 1)

    # -----------------------------------------------------------------------
    # Test 3 — Vector has the expected dimension
    # -----------------------------------------------------------------------
    def test_embed_text_correct_dimension(self) -> None:
        """embed_text() returns a vector of the expected dimension."""
        vector = self.embedder.embed_text("Broken access control vulnerability.")
        self.assertEqual(vector.shape[0], _MOCK_DIM)
        self.assertEqual(vector.shape[0], self.embedder.embedding_dim)

    # -----------------------------------------------------------------------
    # Test 4 — Vector dtype is float32
    # -----------------------------------------------------------------------
    def test_embed_text_dtype_is_float32(self) -> None:
        """embed_text() returns a float32 array (FAISS-compatible)."""
        vector = self.embedder.embed_text("OWASP Top 10 security risks.")
        self.assertEqual(vector.dtype, np.float32)

    # -----------------------------------------------------------------------
    # Test 5 — Batch embedding returns one vector per input chunk
    # -----------------------------------------------------------------------
    def test_embed_chunks_returns_one_vector_per_chunk(self) -> None:
        """embed_chunks() returns exactly N vectors for N input chunks."""
        texts = [
            "SQL injection attack",
            "Cross-site scripting",
            "Broken authentication",
        ]
        chunks = _make_chunks(texts)
        result = self.embedder.embed_chunks(chunks)

        self.assertEqual(result.embeddings.shape[0], len(chunks))
        self.assertEqual(result.embeddings.shape[1], _MOCK_DIM)

    # -----------------------------------------------------------------------
    # Test 6 — Input ordering is preserved
    # -----------------------------------------------------------------------
    def test_embed_chunks_preserves_input_order(self) -> None:
        """chunks[i] corresponds to embeddings[i] — order is never shuffled."""
        texts = [f"Security text number {i}" for i in range(5)]
        chunks = _make_chunks(texts)
        result = self.embedder.embed_chunks(chunks)

        self.assertEqual(len(result.chunks), len(chunks))
        for i, (original, returned) in enumerate(zip(chunks, result.chunks)):
            self.assertEqual(
                original.chunk_id,
                returned.chunk_id,
                msg=f"Ordering mismatch at index {i}",
            )

    # -----------------------------------------------------------------------
    # Test 7 — Empty batch returns correct shape
    # -----------------------------------------------------------------------
    def test_embed_chunks_empty_list_returns_correct_shape(self) -> None:
        """embed_chunks([]) returns EmbeddedChunks with shape (0, embedding_dim)."""
        result = self.embedder.embed_chunks([])
        self.assertEqual(result.embeddings.shape, (0, _MOCK_DIM))
        self.assertEqual(len(result.chunks), 0)
        self.assertEqual(result.embeddings.dtype, np.float32)

    # -----------------------------------------------------------------------
    # Test 8 — Empty text raises ValueError
    # -----------------------------------------------------------------------
    def test_embed_text_empty_string_raises_value_error(self) -> None:
        """embed_text('') must raise ValueError — not silently produce a vector."""
        with self.assertRaises(ValueError):
            self.embedder.embed_text("")

    def test_embed_text_whitespace_only_raises_value_error(self) -> None:
        """embed_text('   ') must raise ValueError — whitespace is not meaningful."""
        with self.assertRaises(ValueError):
            self.embedder.embed_text("   ")

    def test_embed_text_newline_only_raises_value_error(self) -> None:
        """embed_text('\\n') must raise ValueError."""
        with self.assertRaises(ValueError):
            self.embedder.embed_text("\n")

    # -----------------------------------------------------------------------
    # Test 9 — Multiple calls reuse the same backend (no reload)
    # -----------------------------------------------------------------------
    def test_multiple_calls_reuse_backend(self) -> None:
        """Two embed_chunks() calls on the same EmbeddingModel reuse the backend."""
        chunks1 = _make_chunks(["XSS vulnerability"], document_id="doc-a")
        chunks2 = _make_chunks(["CSRF attack pattern"], document_id="doc-b")

        initial_call_count = self.mock_backend.encode_call_count

        self.embedder.embed_chunks(chunks1)
        self.embedder.embed_chunks(chunks2)

        # encode() was called once per embed_chunks() call (2 total new calls).
        new_calls = self.mock_backend.encode_call_count - initial_call_count
        self.assertEqual(new_calls, 2, "Expected exactly 2 new encode() calls.")

        # Critically: there is still only ONE backend instance.
        # If the model were reloaded, a new MockBackend would be created.
        self.assertIs(self.embedder._backend, self.mock_backend)

    # -----------------------------------------------------------------------
    # Test 10 — Embeddings are L2-normalised
    # -----------------------------------------------------------------------
    def test_embed_text_is_unit_vector(self) -> None:
        """embed_text() returns a unit vector (L2 norm ≈ 1.0)."""
        vector = self.embedder.embed_text("Path traversal attack.")
        norm = float(np.linalg.norm(vector))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_embed_chunks_rows_are_unit_vectors(self) -> None:
        """All rows of embed_chunks() result are unit vectors."""
        chunks = _make_chunks([
            "Buffer overflow exploit",
            "Remote code execution",
            "Privilege escalation",
        ])
        result = self.embedder.embed_chunks(chunks)
        for i in range(result.embeddings.shape[0]):
            norm = float(np.linalg.norm(result.embeddings[i]))
            self.assertAlmostEqual(norm, 1.0, places=5, msg=f"Row {i} is not normalised.")

    def test_normalise_vector_with_norm_below_one(self) -> None:
        """Vectors with norm < 1.0 are properly normalized to unit length (norm == 1.0)."""
        # Vector with small elements: norm is sqrt(0.01 + 0.04) = sqrt(0.05) ≈ 0.2236 < 1.0
        small_vec = np.array([[0.1, 0.2, 0.0, 0.0]], dtype=np.float32)
        self.assertLess(float(np.linalg.norm(small_vec)), 1.0)

        normalised = EmbeddingModel._normalise(small_vec)
        norm = float(np.linalg.norm(normalised))
        self.assertAlmostEqual(norm, 1.0, places=5)
        self.assertEqual(normalised.dtype, np.float32)

    def test_normalise_zero_vector_handled_safely(self) -> None:
        """All-zero vectors are safely handled with epsilon without producing NaNs."""
        zero_vec = np.zeros((1, 4), dtype=np.float32)
        normalised = EmbeddingModel._normalise(zero_vec)
        self.assertTrue(np.all(normalised == 0.0))
        self.assertFalse(np.isnan(normalised).any())
        self.assertEqual(normalised.dtype, np.float32)


    # -----------------------------------------------------------------------
    # Test 11 — Same input produces equivalent embeddings
    # -----------------------------------------------------------------------
    def test_same_input_produces_same_embedding(self) -> None:
        """The same text always produces the same vector (determinism)."""
        text = "Injection vulnerabilities allow attackers to send hostile data."
        v1 = self.embedder.embed_text(text)
        v2 = self.embedder.embed_text(text)
        np.testing.assert_array_almost_equal(v1, v2, decimal=5)

    def test_embed_chunks_deterministic(self) -> None:
        """embed_chunks() with the same input always returns the same matrix."""
        chunks = _make_chunks(["OWASP A01: Broken Access Control"])
        r1 = self.embedder.embed_chunks(chunks)
        r2 = self.embedder.embed_chunks(chunks)
        np.testing.assert_array_almost_equal(r1.embeddings, r2.embeddings, decimal=5)

    # -----------------------------------------------------------------------
    # Test 12 — Semantic similarity (mock: same text → similarity 1.0)
    # -----------------------------------------------------------------------
    def test_same_text_has_similarity_one(self) -> None:
        """For the mock backend, the same text twice yields cosine similarity 1.0."""
        text = "SQL injection attack on login form"
        v1 = self.embedder.embed_text(text)
        v2 = self.embedder.embed_text(text)
        sim = _cosine_sim(v1, v2)
        self.assertAlmostEqual(sim, 1.0, places=5)

    # -----------------------------------------------------------------------
    # Test 13 — EmbeddedChunks invariant
    # -----------------------------------------------------------------------
    def test_embedded_chunks_invariant_mismatch_raises(self) -> None:
        """EmbeddedChunks raises ValueError if len(chunks) != embeddings.shape[0]."""
        chunks = _make_chunks(["text a", "text b"])
        wrong_matrix = np.zeros((5, _MOCK_DIM), dtype=np.float32)
        with self.assertRaises(ValueError):
            EmbeddedChunks(chunks=chunks, embeddings=wrong_matrix, model_name="mock")

    def test_embedded_chunks_embedding_dim_property(self) -> None:
        """EmbeddedChunks.embedding_dim matches the matrix second dimension."""
        chunks = _make_chunks(["access control"])
        result = self.embedder.embed_chunks(chunks)
        self.assertEqual(result.embedding_dim, _MOCK_DIM)

    # -----------------------------------------------------------------------
    # Test 14 — EmbeddingModel.embedding_dim matches actual vector length
    # -----------------------------------------------------------------------
    def test_embedding_dim_property_matches_vector_length(self) -> None:
        """EmbeddingModel.embedding_dim equals the length of a produced vector."""
        vector = self.embedder.embed_text("Cybersecurity knowledge base.")
        self.assertEqual(self.embedder.embedding_dim, len(vector))

    # -----------------------------------------------------------------------
    # Additional edge cases
    # -----------------------------------------------------------------------
    def test_embed_chunks_dtype_is_float32(self) -> None:
        """embed_chunks() result dtype must be float32."""
        chunks = _make_chunks(["MITRE ATT&CK technique T1059"])
        result = self.embedder.embed_chunks(chunks)
        self.assertEqual(result.embeddings.dtype, np.float32)

    def test_embed_chunks_model_name_preserved(self) -> None:
        """EmbeddedChunks.model_name matches the EmbeddingModel's model_name."""
        chunks = _make_chunks(["test chunk"])
        result = self.embedder.embed_chunks(chunks)
        self.assertEqual(result.model_name, self.embedder.model_name)

    def test_embed_chunks_single_chunk(self) -> None:
        """embed_chunks() works correctly with exactly one chunk."""
        chunks = _make_chunks(["A single chunk of security text."])
        result = self.embedder.embed_chunks(chunks)
        self.assertEqual(result.embeddings.shape, (1, _MOCK_DIM))
        self.assertEqual(len(result.chunks), 1)

    def test_embed_chunks_returns_defensive_copy_of_chunks(self) -> None:
        """Modifying the original list after embed_chunks() does not affect result."""
        texts = ["first chunk", "second chunk"]
        chunks = _make_chunks(texts)
        result = self.embedder.embed_chunks(chunks)
        original_ids = [c.chunk_id for c in result.chunks]

        # Mutate the original list.
        chunks.clear()

        returned_ids = [c.chunk_id for c in result.chunks]
        self.assertEqual(original_ids, returned_ids)

    def test_embed_text_tab_only_raises_value_error(self) -> None:
        """embed_text() with tab-only string raises ValueError."""
        with self.assertRaises(ValueError):
            self.embedder.embed_text("\t\t")

    def test_convenience_function_embed_chunks(self) -> None:
        """Module-level embed_chunks() convenience function works end-to-end."""
        chunks = _make_chunks(["SQL injection", "XSS attack"])
        # We cannot inject a mock backend through the convenience function,
        # so we patch SentenceTransformerBackend.
        mock_backend = MockBackend(dim=_MOCK_DIM)
        with patch(
            "app.rag.embeddings.SentenceTransformerBackend",
            return_value=mock_backend,
        ):
            result = embed_chunks(chunks)
        self.assertEqual(result.embeddings.shape[0], 2)


# ===========================================================================
# Integration Tests — real model, cached locally
# ===========================================================================

def _real_model_available() -> bool:
    """Return True only if the cached model is available without network access."""
    try:
        import os
        cache_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
        )
        return os.path.isdir(cache_dir)
    except Exception:
        return False


_SKIP_INTEGRATION = not _real_model_available()
_SKIP_REASON = (
    "Skipping integration tests: all-MiniLM-L6-v2 not found in local cache. "
    "Run once with network access to download and cache the model."
)

_REAL_DIM = 384  # Expected dimension for all-MiniLM-L6-v2


@unittest.skipIf(_SKIP_INTEGRATION, _SKIP_REASON)
class TestEmbeddingModelIntegration(unittest.TestCase):
    """
    Integration tests using the real sentence-transformers model.

    Only runs when the locally-cached model is available.
    Uses setUpClass so the model is loaded once for the entire class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Load the real model once for all integration tests."""
        cls.embedder = EmbeddingModel()

    # -----------------------------------------------------------------------
    # Test 15 — Real model initialises with correct dimension
    # -----------------------------------------------------------------------
    def test_real_model_embedding_dim(self) -> None:
        """all-MiniLM-L6-v2 must produce 384-dimensional embeddings."""
        self.assertEqual(self.embedder.embedding_dim, _REAL_DIM)

    # -----------------------------------------------------------------------
    # Test 16 — Real embed_text returns float32 unit vector of dim 384
    # -----------------------------------------------------------------------
    def test_real_embed_text_shape_and_dtype(self) -> None:
        """Real embed_text() returns a (384,) float32 unit vector."""
        vector = self.embedder.embed_text("SQL injection attack vector.")
        self.assertEqual(vector.shape, (_REAL_DIM,))
        self.assertEqual(vector.dtype, np.float32)
        norm = float(np.linalg.norm(vector))
        self.assertAlmostEqual(norm, 1.0, places=5)

    # -----------------------------------------------------------------------
    # Test 17 — Real embed_chunks preserves order and shape
    # -----------------------------------------------------------------------
    def test_real_embed_chunks_shape_and_order(self) -> None:
        """Real embed_chunks() returns (N, 384) float32 matrix in input order."""
        texts = [
            "Broken Access Control allows privilege escalation.",
            "Cryptographic failures expose sensitive data.",
            "Injection attacks manipulate queries.",
        ]
        chunks = _make_chunks(texts)
        result = self.embedder.embed_chunks(chunks)

        self.assertEqual(result.embeddings.shape, (3, _REAL_DIM))
        self.assertEqual(result.embeddings.dtype, np.float32)

        for i, (chunk, returned) in enumerate(zip(chunks, result.chunks)):
            self.assertEqual(
                chunk.chunk_id,
                returned.chunk_id,
                msg=f"Order mismatch at index {i}",
            )

    def test_real_empty_chunks_shape(self) -> None:
        """Real model: embed_chunks([]) returns shape (0, 384)."""
        result = self.embedder.embed_chunks([])
        self.assertEqual(result.embeddings.shape, (0, _REAL_DIM))

    # -----------------------------------------------------------------------
    # Test 18 — Semantic similarity test with real model
    # -----------------------------------------------------------------------
    def test_real_semantic_similarity(self) -> None:
        """
        Semantically related pair scores higher than an obviously unrelated pair.

        Related: two sentences both about SQL injection.
        Unrelated: SQL injection vs. a non-security phrase about weather.
        """
        related_a = "SQL injection allows attackers to manipulate database queries."
        related_b = "Parameterised queries prevent SQL injection attacks."
        unrelated  = "The weather today is warm and sunny with clear blue skies."

        v_related_a = self.embedder.embed_text(related_a)
        v_related_b = self.embedder.embed_text(related_b)
        v_unrelated = self.embedder.embed_text(unrelated)

        sim_related   = _cosine_sim(v_related_a, v_related_b)
        sim_unrelated = _cosine_sim(v_related_a, v_unrelated)

        self.assertGreater(
            sim_related,
            sim_unrelated,
            msg=(
                f"Expected related pair similarity ({sim_related:.4f}) > "
                f"unrelated pair similarity ({sim_unrelated:.4f})."
            ),
        )

    def test_real_same_input_determinism(self) -> None:
        """Real model: same text produces the same vector on repeated calls."""
        text = "OWASP Top 10: A03 Injection vulnerabilities."
        v1 = self.embedder.embed_text(text)
        v2 = self.embedder.embed_text(text)
        np.testing.assert_array_almost_equal(v1, v2, decimal=5)

    def test_real_model_name(self) -> None:
        """Real EmbeddingModel stores the correct default model name."""
        self.assertEqual(self.embedder.model_name, DEFAULT_MODEL_NAME)

    def test_real_rows_are_unit_vectors(self) -> None:
        """All rows in real embed_chunks() result are L2-normalised."""
        chunks = _make_chunks([
            "Authentication bypass via weak credentials.",
            "Insecure direct object reference.",
            "Server-side request forgery (SSRF).",
            "Security misconfiguration in cloud storage.",
        ])
        result = self.embedder.embed_chunks(chunks)
        for i in range(result.embeddings.shape[0]):
            norm = float(np.linalg.norm(result.embeddings[i]))
            self.assertAlmostEqual(norm, 1.0, places=5, msg=f"Row {i} not normalised.")


# ===========================================================================
# SentenceTransformerBackend unit tests (import-level, no model loading)
# ===========================================================================

class TestSentenceTransformerBackendImport(unittest.TestCase):
    """Tests that verify import-time behaviour without loading the model."""

    def test_missing_sentence_transformers_raises_import_error(self) -> None:
        """
        SentenceTransformerBackend raises ImportError if sentence-transformers
        is not installed.
        """
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with self.assertRaises((ImportError, TypeError)):
                SentenceTransformerBackend(model_name="nonexistent/model")


class TestRAGPackageExports(unittest.TestCase):
    """Tests that verify app.rag exposes all expected embedding symbols."""

    def test_app_rag_exports_embedding_symbols(self) -> None:
        """Verify symbols can be imported directly from app.rag."""
        import app.rag as rag

        expected = [
            "DEFAULT_MODEL_NAME",
            "DEFAULT_BATCH_SIZE",
            "EmbeddingBackend",
            "SentenceTransformerBackend",
            "EmbeddedChunks",
            "EmbeddingModel",
            "embed_chunks",
        ]
        for name in expected:
            self.assertTrue(hasattr(rag, name), f"app.rag missing export: {name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

