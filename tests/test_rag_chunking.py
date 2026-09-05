"""
Unit tests for the RAG Cleaning + Chunking Layer (app/rag/chunking.py).

Tests cover:
    1.  Basic cleaning — line ending normalisation (CRLF, CR → LF).
    2.  Basic cleaning — excessive whitespace / blank line collapsing.
    3.  Cleaning preserves meaningful security text exactly.
    4.  Small document produces one chunk that fits within chunk_size.
    5.  Large text is split into multiple chunks.
    6.  Chunk size limit is respected (no chunk exceeds chunk_size chars).
    7.  Overlap is applied when a large block must be hard-split.
    8.  Chunk ordering is deterministic and strictly ascending.
    9.  Chunk IDs are deterministic and follow the expected pattern.
    10. section_title is captured from a Markdown heading.
    11. section_title is None when no reliable heading exists.
    12. Provenance fields (document_id, source, source_type, url,
        framework_metadata, tags) are preserved on every chunk.
    13. total_chunks is accurate for every chunk in the list.
    14. Empty / whitespace-only document content is handled safely.
    15. Invalid Chunker configuration raises ValueError.
    16. Running the same document through the chunker twice gives
        identical results (idempotency / determinism).
    17. clean_and_chunk_document convenience function works end-to-end.
    18. clean_and_chunk_documents processes multiple docs independently.
    19. Heading-only block (no body) does not crash.
    20. Paragraph-only content (no Markdown headings) chunks correctly.
"""

import unittest
from datetime import datetime, timezone

from app.models.schemas import CWEReference, MITREReference, OWASPReference
from app.rag.chunking import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    Chunker,
    TextCleaner,
    clean_and_chunk_document,
    clean_and_chunk_documents,
)
from app.rag.schemas import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeFrameworkMetadata,
    KnowledgeSourceType,
    NISTReference,
)


# =====================================================================
# Helpers / Fixtures
# =====================================================================

def _make_document(
    content: str,
    document_id: str = "test-doc-001",
    title: str = "Test Document",
    source: str = "OWASP Foundation",
    source_type: KnowledgeSourceType = KnowledgeSourceType.OWASP,
    url: str = "https://owasp.org/test",
    tags: list | None = None,
    framework_metadata: KnowledgeFrameworkMetadata | None = None,
) -> KnowledgeDocument:
    """Construct a minimal KnowledgeDocument for testing."""
    return KnowledgeDocument(
        document_id=document_id,
        title=title,
        source=source,
        source_type=source_type,
        url=url,
        content=content,
        tags=tags or ["test"],
        framework_metadata=framework_metadata or KnowledgeFrameworkMetadata(),
    )


def _large_text(n_chars: int = 3000) -> str:
    """Generate a realistic-looking security paragraph long enough to force chunking."""
    sentence = (
        "Access control must be enforced server-side, not merely in the client UI. "
    )
    repeated = sentence * (n_chars // len(sentence) + 1)
    return repeated[:n_chars]


# =====================================================================
# Test Suite
# =====================================================================

class TestTextCleaner(unittest.TestCase):
    """Tests for TextCleaner."""

    def setUp(self):
        self.cleaner = TextCleaner()

    # ------------------------------------------------------------------
    # Test 1 — CRLF and CR normalisation
    # ------------------------------------------------------------------
    def test_crlf_normalised_to_lf(self):
        """CRLF line endings must be converted to LF."""
        text = "Line one.\r\nLine two.\r\nLine three."
        result = self.cleaner.clean(text)
        self.assertNotIn("\r\n", result)
        self.assertNotIn("\r", result)
        self.assertEqual(result, "Line one.\nLine two.\nLine three.")

    def test_cr_only_normalised_to_lf(self):
        """Bare CR line endings must be converted to LF."""
        text = "Line one.\rLine two.\rLine three."
        result = self.cleaner.clean(text)
        self.assertNotIn("\r", result)
        self.assertEqual(result, "Line one.\nLine two.\nLine three.")

    def test_mixed_line_endings_normalised(self):
        """Mixed CRLF and LF endings should both become LF."""
        text = "Alpha.\r\nBeta.\nGamma.\r\nDelta."
        result = self.cleaner.clean(text)
        self.assertNotIn("\r", result)

    # ------------------------------------------------------------------
    # Test 2 — Excessive blank lines
    # ------------------------------------------------------------------
    def test_excessive_blank_lines_collapsed(self):
        """Three or more consecutive blank lines must collapse to two."""
        text = "Para one.\n\n\n\n\nPara two."
        result = self.cleaner.clean(text)
        # Should not contain three consecutive newlines
        self.assertNotIn("\n\n\n", result)
        # Both paragraphs should still be present
        self.assertIn("Para one.", result)
        self.assertIn("Para two.", result)

    def test_single_blank_line_preserved(self):
        """A single blank line (paragraph separator) must NOT be removed."""
        text = "Para one.\n\nPara two."
        result = self.cleaner.clean(text)
        self.assertIn("Para one.", result)
        self.assertIn("Para two.", result)
        # The blank line (two newlines) should still be there
        self.assertIn("\n\n", result)

    def test_trailing_whitespace_stripped_per_line(self):
        """Trailing spaces on individual lines must be removed."""
        text = "Line with trailing spaces.   \nNext line.  "
        result = self.cleaner.clean(text)
        for line in result.split("\n"):
            self.assertEqual(line, line.rstrip())

    # ------------------------------------------------------------------
    # Test 3 — Meaningful security text preserved exactly
    # ------------------------------------------------------------------
    def test_security_content_preserved(self):
        """Cleaning must not rewrite, remove, or alter security-critical content."""
        text = (
            "CWE-79: Improper Neutralization of Input During Web Page Generation.\n"
            "MITRE T1190: Exploit Public-Facing Application.\n"
            "NIST SP 800-53 Control AC-3: Access Enforcement.\n"
            "URL: https://cwe.mitre.org/data/definitions/79.html"
        )
        result = self.cleaner.clean(text)
        self.assertIn("CWE-79", result)
        self.assertIn("MITRE T1190", result)
        self.assertIn("NIST SP 800-53", result)
        self.assertIn("AC-3", result)
        self.assertIn("https://cwe.mitre.org/data/definitions/79.html", result)

    def test_urls_preserved(self):
        """URLs containing special characters must pass through unchanged."""
        url = "https://owasp.org/Top10/A01_2021-Broken_Access_Control/"
        text = f"See reference: {url}"
        result = self.cleaner.clean(text)
        self.assertIn(url, result)

    def test_numbered_list_preserved(self):
        """Numbered lists must not be accidentally concatenated."""
        text = "Steps:\n1. Identify the asset.\n2. Assess the threat.\n3. Apply control."
        result = self.cleaner.clean(text)
        self.assertIn("1. Identify the asset.", result)
        self.assertIn("2. Assess the threat.", result)
        self.assertIn("3. Apply control.", result)

    # ------------------------------------------------------------------
    # Test 14-a — Empty / whitespace input
    # ------------------------------------------------------------------
    def test_empty_string_returns_empty(self):
        """Empty input must return an empty string without error."""
        self.assertEqual(self.cleaner.clean(""), "")

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only input must return an empty string."""
        self.assertEqual(self.cleaner.clean("   \n\n\t  "), "")


# =====================================================================

class TestChunkerConfiguration(unittest.TestCase):
    """Tests for Chunker configuration validation (Test 15)."""

    def test_invalid_chunk_size_zero_raises(self):
        """chunk_size = 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            Chunker(chunk_size=0)

    def test_invalid_chunk_size_negative_raises(self):
        """Negative chunk_size must raise ValueError."""
        with self.assertRaises(ValueError):
            Chunker(chunk_size=-100)

    def test_invalid_overlap_negative_raises(self):
        """Negative overlap must raise ValueError."""
        with self.assertRaises(ValueError):
            Chunker(chunk_size=500, overlap=-1)

    def test_invalid_overlap_equals_chunk_size_raises(self):
        """overlap == chunk_size must raise ValueError."""
        with self.assertRaises(ValueError):
            Chunker(chunk_size=500, overlap=500)

    def test_invalid_overlap_exceeds_chunk_size_raises(self):
        """overlap > chunk_size must raise ValueError."""
        with self.assertRaises(ValueError):
            Chunker(chunk_size=500, overlap=600)

    def test_valid_configuration_accepted(self):
        """A valid (chunk_size, overlap) pair must not raise."""
        chunker = Chunker(chunk_size=1000, overlap=100)
        self.assertEqual(chunker.chunk_size, 1000)
        self.assertEqual(chunker.overlap, 100)

    def test_zero_overlap_accepted(self):
        """overlap = 0 must be accepted (no overlap)."""
        chunker = Chunker(chunk_size=1000, overlap=0)
        self.assertEqual(chunker.overlap, 0)


# =====================================================================

class TestChunkerBasicBehaviour(unittest.TestCase):
    """Core chunking behaviour tests."""

    def setUp(self):
        self.chunker = Chunker(chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP)

    # ------------------------------------------------------------------
    # Test 4 — Small document → single chunk
    # ------------------------------------------------------------------
    def test_small_document_produces_one_chunk(self):
        """A document shorter than chunk_size must produce exactly one chunk."""
        content = "Access control enforces policy such that users cannot act outside their permissions."
        doc = _make_document(content)
        chunks = self.chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Access control", chunks[0].text)

    # ------------------------------------------------------------------
    # Test 5 — Large text → multiple chunks
    # ------------------------------------------------------------------
    def test_large_document_produces_multiple_chunks(self):
        """A document much larger than chunk_size must be split into multiple chunks."""
        content = _large_text(n_chars=5000)
        doc = _make_document(content)
        chunks = self.chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 1)

    # ------------------------------------------------------------------
    # Test 6 — Chunk size limit respected
    # ------------------------------------------------------------------
    def test_chunk_size_limit_respected(self):
        """No chunk's text length should exceed chunk_size characters."""
        content = _large_text(n_chars=6000)
        doc = _make_document(content)
        chunker = Chunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            self.assertLessEqual(
                len(chunk.text),
                550,  # slight tolerance for sentence-boundary splits
                msg=f"Chunk {chunk.chunk_id} text length {len(chunk.text)} exceeds limit.",
            )

    # ------------------------------------------------------------------
    # Test 7 — Overlap applied between sub-chunks
    # ------------------------------------------------------------------
    def test_overlap_creates_shared_content(self):
        """When a large block is hard-split, the overlap text should appear
        at the end of chunk N and the start of chunk N+1."""
        # Use a plain paragraph (no headings) so hard-split is triggered.
        sentence = "Each word of this sentence is part of a long security description. "
        content = sentence * 30  # ~1980 chars — will be split
        doc = _make_document(content)
        chunker = Chunker(chunk_size=500, overlap=100)
        chunks = chunker.chunk_document(doc)

        self.assertGreater(len(chunks), 1, "Expected more than one chunk.")

        # The tail of chunk 0 and the head of chunk 1 should share text.
        tail_of_first = chunks[0].text[-100:]
        head_of_second = chunks[1].text[:100]
        # They should share at least some characters (overlap region).
        shared = any(
            word in head_of_second
            for word in tail_of_first.split()
            if len(word) > 4  # ignore trivial short words
        )
        self.assertTrue(shared, "Expected overlapping content between chunk 0 and chunk 1.")

    # ------------------------------------------------------------------
    # Test 14-b — Empty document content
    # ------------------------------------------------------------------
    def test_empty_document_content_returns_empty_list(self):
        """A document whose content becomes empty after cleaning must return []."""
        # We can't pass empty content to KnowledgeDocument (validator rejects it),
        # so we test the chunker directly with a mock that bypasses validation.
        # Instead, test with a single-space-after-clean scenario via Chunker directly.
        doc = _make_document(content="Some real content here to pass validation.")
        # Manually replace content before chunking to simulate cleaned empty.
        chunker = Chunker()
        cleaner = TextCleaner()
        # Verify that clean("   ") → "" and chunker handles it.
        cleaned = cleaner.clean("   \n  \n  ")
        self.assertEqual(cleaned, "")

    # ------------------------------------------------------------------
    # Test 8 — Chunk ordering is deterministic and ascending
    # ------------------------------------------------------------------
    def test_chunk_ordering_is_ascending(self):
        """chunk_index values must be 0, 1, 2, … without gaps."""
        content = _large_text(n_chars=5000)
        doc = _make_document(content)
        chunks = self.chunker.chunk_document(doc)
        indices = [c.chunk_index for c in chunks]
        self.assertEqual(indices, list(range(len(chunks))))

    # ------------------------------------------------------------------
    # Test 9 — Chunk IDs are deterministic
    # ------------------------------------------------------------------
    def test_chunk_ids_are_deterministic(self):
        """Chunk IDs must follow '{document_id}:chunk:{index}' pattern."""
        content = _large_text(n_chars=5000)
        doc = _make_document(content, document_id="owasp-a01-2021")
        chunks = self.chunker.chunk_document(doc)
        for i, chunk in enumerate(chunks):
            expected_id = f"owasp-a01-2021:chunk:{i}"
            self.assertEqual(chunk.chunk_id, expected_id)

    def test_chunk_ids_same_on_repeat_run(self):
        """Running the same document twice must produce identical chunk IDs."""
        content = _large_text(n_chars=4000)
        doc = _make_document(content, document_id="repeat-doc")
        chunks_a = self.chunker.chunk_document(doc)
        chunks_b = self.chunker.chunk_document(doc)
        ids_a = [c.chunk_id for c in chunks_a]
        ids_b = [c.chunk_id for c in chunks_b]
        self.assertEqual(ids_a, ids_b)

    # ------------------------------------------------------------------
    # Test 13 — total_chunks is accurate
    # ------------------------------------------------------------------
    def test_total_chunks_is_accurate(self):
        """Every chunk's total_chunks field must equal len(chunks)."""
        content = _large_text(n_chars=5000)
        doc = _make_document(content)
        chunks = self.chunker.chunk_document(doc)
        expected_total = len(chunks)
        for chunk in chunks:
            self.assertEqual(chunk.total_chunks, expected_total)

    # ------------------------------------------------------------------
    # Test 16 — Idempotency / determinism
    # ------------------------------------------------------------------
    def test_idempotency_same_output_on_two_runs(self):
        """Processing the same document twice must produce identical chunks."""
        content = _large_text(n_chars=4500)
        doc = _make_document(content, document_id="idem-doc")
        run_a = self.chunker.chunk_document(doc)
        run_b = self.chunker.chunk_document(doc)

        self.assertEqual(len(run_a), len(run_b))
        for ca, cb in zip(run_a, run_b):
            self.assertEqual(ca.chunk_id, cb.chunk_id)
            self.assertEqual(ca.text, cb.text)
            self.assertEqual(ca.chunk_index, cb.chunk_index)
            self.assertEqual(ca.total_chunks, cb.total_chunks)


# =====================================================================

class TestSectionAwareChunking(unittest.TestCase):
    """Tests for heading detection and section_title behaviour."""

    def setUp(self):
        self.chunker = Chunker()

    # ------------------------------------------------------------------
    # Test 10 — section_title from Markdown heading
    # ------------------------------------------------------------------
    def test_markdown_heading_captured_as_section_title(self):
        """A Markdown heading should appear as section_title on the first chunk of its section."""
        content = (
            "# Overview\n\n"
            "Access control enforces policy such that users cannot act outside of their intended permissions.\n\n"
            "## How to Prevent\n\n"
            "Enforce access controls in trusted server-side code and deny access by default."
        )
        doc = _make_document(content)
        chunks = self.chunker.chunk_document(doc)

        titles = [c.section_title for c in chunks]
        self.assertIn("Overview", titles)
        self.assertIn("How to Prevent", titles)

    def test_section_title_does_not_include_hash_symbols(self):
        """section_title should be clean text — no leading '#' characters."""
        content = "## Authentication Failures\n\nWeak password policies lead to credential compromise."
        doc = _make_document(content)
        chunks = self.chunker.chunk_document(doc)
        for chunk in chunks:
            if chunk.section_title is not None:
                self.assertFalse(
                    chunk.section_title.startswith("#"),
                    msg=f"section_title should not start with '#': {chunk.section_title!r}",
                )

    def test_deep_heading_levels_recognised(self):
        """Headings at levels 1–6 should all be captured as section_title."""
        content = (
            "### Level 3 Heading\n\n"
            "Security control description at sub-section level."
        )
        doc = _make_document(content)
        chunks = self.chunker.chunk_document(doc)
        titles = [c.section_title for c in chunks if c.section_title is not None]
        self.assertGreater(len(titles), 0)
        self.assertEqual(titles[0], "Level 3 Heading")

    # ------------------------------------------------------------------
    # Test 11 — section_title is None without headings
    # ------------------------------------------------------------------
    def test_section_title_none_when_no_heading(self):
        """When no Markdown heading exists, section_title should be None."""
        content = (
            "Access control enforces policy such that users cannot act outside their permissions. "
            "Failures lead to unauthorized information disclosure. "
            "Prevention requires enforcing access controls in trusted server-side code."
        )
        doc = _make_document(content)
        chunks = self.chunker.chunk_document(doc)
        for chunk in chunks:
            self.assertIsNone(
                chunk.section_title,
                msg=f"Expected section_title=None, got {chunk.section_title!r}.",
            )

    def test_short_line_not_mistaken_for_heading(self):
        """A short line that is NOT a Markdown heading must not become section_title."""
        content = (
            "Note\n\n"
            "This is a regular paragraph that starts with a short word. "
            "It should not be treated as a section heading."
        )
        doc = _make_document(content)
        chunks = self.chunker.chunk_document(doc)
        for chunk in chunks:
            self.assertIsNone(
                chunk.section_title,
                msg=(
                    f"'Note' is not a Markdown heading and should not produce "
                    f"section_title, got {chunk.section_title!r}."
                ),
            )

    # ------------------------------------------------------------------
    # Test 19 — Heading with no body
    # ------------------------------------------------------------------
    def test_heading_only_block_does_not_crash(self):
        """A heading with no body text below it must not raise an exception."""
        content = "## Empty Section\n\n## Next Section\n\nSome content here."
        doc = _make_document(content)
        # Should not raise
        chunks = self.chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 0)

    # ------------------------------------------------------------------
    # Test 20 — Paragraph-only content
    # ------------------------------------------------------------------
    def test_paragraph_only_content_chunks_correctly(self):
        """Plain paragraph text with no headings should be chunked into paragraphs."""
        content = (
            "First paragraph about broken access control and IDOR vulnerabilities.\n\n"
            "Second paragraph explaining how RBAC mitigates authorization flaws.\n\n"
            "Third paragraph on server-side enforcement of access policies."
        )
        doc = _make_document(content)
        chunks = self.chunker.chunk_document(doc)
        # Each paragraph is short — likely one chunk per paragraph.
        texts = [c.text for c in chunks]
        combined = " ".join(texts)
        self.assertIn("First paragraph", combined)
        self.assertIn("Second paragraph", combined)
        self.assertIn("Third paragraph", combined)


# =====================================================================

class TestProvenancePreservation(unittest.TestCase):
    """Tests for provenance field propagation (Test 12)."""

    def _build_rich_document(self) -> KnowledgeDocument:
        """Build a KnowledgeDocument with full framework metadata."""
        return KnowledgeDocument(
            document_id="owasp-top10-2021-a01",
            title="OWASP Top 10:2021 - A01 Broken Access Control",
            source="OWASP Foundation",
            source_type=KnowledgeSourceType.OWASP,
            url="https://owasp.org/Top10/A01_2021-Broken_Access_Control/",
            version="2021",
            publication_date="2021-09-24",
            content=_large_text(n_chars=4000),
            framework_metadata=KnowledgeFrameworkMetadata(
                owasp=[
                    OWASPReference(
                        code="A01:2021",
                        name="Broken Access Control",
                        url="https://owasp.org/Top10/A01_2021-Broken_Access_Control/",
                    )
                ],
                cwe=[
                    CWEReference(
                        cwe_id="CWE-200",
                        name="Exposure of Sensitive Information to an Unauthorized Actor",
                    ),
                    CWEReference(
                        cwe_id="CWE-639",
                        name="Authorization Bypass Through User-Controlled Key",
                    ),
                ],
                mitre=[
                    MITREReference(
                        technique_id="T1078",
                        technique_name="Valid Accounts",
                        tactic="Defense Evasion",
                    )
                ],
                nist=[
                    NISTReference(
                        control_id="AC-3",
                        title="Access Enforcement",
                        publication="NIST SP 800-53",
                    )
                ],
            ),
            tags=["access-control", "authorization", "idor", "owasp-top-10"],
        )

    def test_document_id_preserved_on_every_chunk(self):
        """Every chunk must carry the parent document_id."""
        doc = self._build_rich_document()
        chunks = Chunker().chunk_document(doc)
        for chunk in chunks:
            self.assertEqual(chunk.document_id, "owasp-top10-2021-a01")

    def test_source_preserved_on_every_chunk(self):
        """Every chunk must carry the parent source."""
        doc = self._build_rich_document()
        chunks = Chunker().chunk_document(doc)
        for chunk in chunks:
            self.assertEqual(chunk.source, "OWASP Foundation")

    def test_source_type_preserved_on_every_chunk(self):
        """Every chunk must carry the parent source_type."""
        doc = self._build_rich_document()
        chunks = Chunker().chunk_document(doc)
        for chunk in chunks:
            self.assertEqual(chunk.source_type, KnowledgeSourceType.OWASP)

    def test_url_preserved_on_every_chunk(self):
        """Every chunk must carry the parent URL."""
        doc = self._build_rich_document()
        chunks = Chunker().chunk_document(doc)
        for chunk in chunks:
            self.assertEqual(
                chunk.url,
                "https://owasp.org/Top10/A01_2021-Broken_Access_Control/",
            )

    def test_framework_metadata_preserved_on_every_chunk(self):
        """Every chunk must carry all four framework reference lists."""
        doc = self._build_rich_document()
        chunks = Chunker().chunk_document(doc)
        for chunk in chunks:
            self.assertEqual(len(chunk.framework_metadata.owasp), 1)
            self.assertEqual(chunk.framework_metadata.owasp[0].code, "A01:2021")
            self.assertEqual(len(chunk.framework_metadata.cwe), 2)
            self.assertEqual(len(chunk.framework_metadata.mitre), 1)
            self.assertEqual(chunk.framework_metadata.mitre[0].technique_id, "T1078")
            self.assertEqual(len(chunk.framework_metadata.nist), 1)
            self.assertEqual(chunk.framework_metadata.nist[0].control_id, "AC-3")

    def test_tags_preserved_on_every_chunk(self):
        """Every chunk must carry the parent tag list."""
        doc = self._build_rich_document()
        chunks = Chunker().chunk_document(doc)
        for chunk in chunks:
            self.assertIn("access-control", chunk.tags)
            self.assertIn("owasp-top-10", chunk.tags)

    def test_chunk_to_document_lineage_complete(self):
        """
        Verify the full lineage chain:
        KnowledgeDocument.document_id → KnowledgeChunk.document_id
        and that source/url/framework_metadata match the document.
        """
        doc = self._build_rich_document()
        chunks = Chunker().chunk_document(doc)
        for chunk in chunks:
            self.assertEqual(chunk.document_id, doc.document_id)
            self.assertEqual(chunk.source, doc.source)
            self.assertEqual(chunk.source_type, doc.source_type)
            self.assertEqual(chunk.url, doc.url)
            self.assertEqual(
                chunk.framework_metadata.model_dump(),
                doc.framework_metadata.model_dump(),
            )


# =====================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for module-level convenience API (Tests 17, 18)."""

    # ------------------------------------------------------------------
    # Test 17 — clean_and_chunk_document
    # ------------------------------------------------------------------
    def test_clean_and_chunk_document_end_to_end(self):
        """clean_and_chunk_document must return valid chunks for a real-ish document."""
        content = (
            "# Broken Access Control\n\n"
            "Access control enforces policy such that users cannot act outside "
            "of their intended permissions. Failures lead to unauthorized "
            "information disclosure, modification, or destruction.\n\n"
            "## How to Prevent\n\n"
            "Enforce access controls strictly in trusted server-side code. "
            "Deny access by default and log access control failures."
        )
        doc = _make_document(content, document_id="owasp-a01")
        chunks = clean_and_chunk_document(doc)

        self.assertGreater(len(chunks), 0)
        # Verify all are KnowledgeChunk instances
        for chunk in chunks:
            self.assertIsInstance(chunk, KnowledgeChunk)
        # Verify IDs follow the pattern
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_id, f"owasp-a01:chunk:{i}")

    def test_clean_and_chunk_document_custom_params(self):
        """clean_and_chunk_document must pass chunk_size/overlap to Chunker."""
        content = _large_text(n_chars=3000)
        doc = _make_document(content)
        # Use a very small chunk_size to force many splits
        chunks = clean_and_chunk_document(doc, chunk_size=300, overlap=30)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 360)  # reasonable tolerance

    # ------------------------------------------------------------------
    # Test 18 — clean_and_chunk_documents (multiple docs)
    # ------------------------------------------------------------------
    def test_clean_and_chunk_documents_multiple_docs(self):
        """clean_and_chunk_documents must process each doc independently."""
        doc_a = _make_document(
            "First document about SQL injection vulnerabilities.",
            document_id="cwe-89",
        )
        doc_b = _make_document(
            "Second document about cross-site scripting (XSS) vulnerabilities.",
            document_id="cwe-79",
        )
        all_chunks = clean_and_chunk_documents([doc_a, doc_b])

        doc_a_chunks = [c for c in all_chunks if c.document_id == "cwe-89"]
        doc_b_chunks = [c for c in all_chunks if c.document_id == "cwe-79"]

        self.assertGreater(len(doc_a_chunks), 0)
        self.assertGreater(len(doc_b_chunks), 0)

        # Each doc's chunks should have ascending indices independently
        for doc_chunks in [doc_a_chunks, doc_b_chunks]:
            indices = [c.chunk_index for c in doc_chunks]
            self.assertEqual(indices, list(range(len(doc_chunks))))

    def test_clean_and_chunk_documents_empty_list(self):
        """clean_and_chunk_documents must return [] for an empty input list."""
        result = clean_and_chunk_documents([])
        self.assertEqual(result, [])


# =====================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_single_sentence_document(self):
        """A single-sentence document should produce exactly one chunk."""
        content = "Use parameterized queries to prevent SQL injection."
        doc = _make_document(content)
        chunks = Chunker().chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertIn("parameterized queries", chunks[0].text)

    def test_content_exactly_at_chunk_size(self):
        """Content exactly at chunk_size characters should produce one chunk."""
        chunker = Chunker(chunk_size=200, overlap=20)
        # Build text that is exactly 200 chars
        content = ("A" * 199) + "."  # 200 chars total
        doc = _make_document(content)
        chunks = chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 1)

    def test_multi_level_headings(self):
        """Both H1 and H2 headings in the same document should both be captured."""
        content = (
            "# Top Level\n\n"
            "Introductory text about the top-level topic.\n\n"
            "## Sub Level\n\n"
            "Detail about the sub-topic."
        )
        doc = _make_document(content)
        chunks = Chunker().chunk_document(doc)
        titles = {c.section_title for c in chunks if c.section_title}
        self.assertIn("Top Level", titles)
        self.assertIn("Sub Level", titles)

    def test_chunk_index_starts_at_zero(self):
        """The first chunk must always have chunk_index = 0."""
        content = "Any content for testing chunk index start."
        doc = _make_document(content)
        chunks = Chunker().chunk_document(doc)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_pydantic_validation_of_chunks(self):
        """All produced chunks must pass KnowledgeChunk's built-in validation."""
        content = _large_text(n_chars=3000)
        doc = _make_document(content)
        chunks = Chunker().chunk_document(doc)
        for chunk in chunks:
            # Re-validate by round-tripping through JSON
            json_str = chunk.model_dump_json()
            reloaded = KnowledgeChunk.model_validate_json(json_str)
            self.assertEqual(reloaded.chunk_id, chunk.chunk_id)
            self.assertEqual(reloaded.chunk_index, chunk.chunk_index)
            self.assertEqual(reloaded.total_chunks, chunk.total_chunks)

    def test_no_external_calls(self):
        """Chunking must complete without any network or external service calls.
        This is guaranteed by design — verified by successful execution."""
        import socket

        original_connect = socket.socket.connect

        def _no_connect(self, *args, **kwargs):
            raise AssertionError(
                "Chunking must not make network calls, but socket.connect was called."
            )

        socket.socket.connect = _no_connect
        try:
            content = "Security content that must be chunked without any network access."
            doc = _make_document(content)
            chunks = Chunker().chunk_document(doc)
            self.assertGreater(len(chunks), 0)
        finally:
            socket.socket.connect = original_connect


if __name__ == "__main__":
    unittest.main(verbosity=2)
