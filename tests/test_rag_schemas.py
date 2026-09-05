"""
Unit tests for Knowledge Base & RAG Data Contracts (app/rag/schemas.py).

Validates:
1. Valid KnowledgeDocument creation and field validation.
2. Rejection of empty or whitespace-only document content and IDs.
3. Valid KnowledgeChunk creation and provenance fields.
4. Source traceability: chunk preserves document_id, source, source_type, and URL.
5. Validation of chunk_index boundaries (ge=0, < total_chunks).
6. Framework metadata representation (OWASP, CWE, MITRE ATT&CK, NIST).
7. JSON serialization and deserialization round-trip integrity.
8. Extensibility: extra fields are safely ignored.
"""

import unittest
from datetime import datetime, timezone
# pyrefly: ignore [missing-import]
from pydantic import ValidationError

from app.models.schemas import CWEReference, MITREReference, OWASPReference
from app.rag.schemas import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeFrameworkMetadata,
    KnowledgeSourceType,
    NISTReference,
)


class TestRAGSchemas(unittest.TestCase):
    """Test suite for Knowledge Base and RAG Pydantic contracts."""

    def test_valid_knowledge_document_creation(self):
        """Test creating a valid KnowledgeDocument with full metadata."""
        doc = KnowledgeDocument(
            document_id="owasp-top10-a01-2021",
            title="A01:2021 - Broken Access Control",
            source="OWASP Foundation",
            source_type=KnowledgeSourceType.OWASP,
            url="https://owasp.org/Top10/A01_2021-Broken_Access_Control/",
            version="2021",
            publication_date="2021-09-24",
            content="Access control enforces policy such that users cannot act outside of their intended permissions.",
            framework_metadata=KnowledgeFrameworkMetadata(
                owasp=[
                    OWASPReference(
                        code="A01:2021",
                        name="Broken Access Control",
                        url="https://owasp.org/Top10/A01_2021-Broken_Access_Control/"
                    )
                ],
                cwe=[
                    CWEReference(
                        cwe_id="CWE-200",
                        name="Exposure of Sensitive Information to an Unauthorized Actor",
                        url="https://cwe.mitre.org/data/definitions/200.html"
                    )
                ],
                mitre=[
                    MITREReference(
                        technique_id="T1078",
                        technique_name="Valid Accounts",
                        tactic="Defense Evasion"
                    )
                ],
                nist=[
                    NISTReference(
                        control_id="AC-3",
                        title="Access Enforcement",
                        publication="NIST SP 800-53",
                        url="https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final"
                    )
                ]
            ),
            tags=["access-control", "authorization", "owasp-top-10"]
        )

        self.assertEqual(doc.document_id, "owasp-top10-a01-2021")
        self.assertEqual(doc.source_type, KnowledgeSourceType.OWASP)
        self.assertEqual(len(doc.framework_metadata.owasp), 1)
        self.assertEqual(len(doc.framework_metadata.nist), 1)
        self.assertEqual(doc.framework_metadata.nist[0].control_id, "AC-3")
        self.assertIsInstance(doc.retrieved_at, datetime)

    def test_document_validation_rejects_empty_content_and_ids(self):
        """Test that empty or whitespace-only content and IDs raise ValidationError."""
        # Empty document_id
        with self.assertRaises(ValidationError):
            KnowledgeDocument(
                document_id="   ",
                title="Valid Title",
                source="OWASP",
                source_type=KnowledgeSourceType.OWASP,
                content="Valid content"
            )

        # Empty title
        with self.assertRaises(ValidationError):
            KnowledgeDocument(
                document_id="doc-01",
                title="",
                source="OWASP",
                source_type=KnowledgeSourceType.OWASP,
                content="Valid content"
            )

        # Empty content
        with self.assertRaises(ValidationError):
            KnowledgeDocument(
                document_id="doc-01",
                title="Valid Title",
                source="OWASP",
                source_type=KnowledgeSourceType.OWASP,
                content="   "
            )

    def test_valid_knowledge_chunk_creation_and_traceability(self):
        """Test KnowledgeChunk creation and verify complete provenance back to parent document."""
        chunk = KnowledgeChunk(
            chunk_id="owasp-a01-chunk-0",
            document_id="owasp-top10-a01-2021",
            text="Access control policies must be enforced server-side, not merely in the client UI.",
            section_title="How to Prevent",
            chunk_index=0,
            total_chunks=3,
            source="OWASP Foundation",
            source_type=KnowledgeSourceType.OWASP,
            url="https://owasp.org/Top10/A01_2021-Broken_Access_Control/",
            framework_metadata=KnowledgeFrameworkMetadata(
                owasp=[OWASPReference(code="A01:2021", name="Broken Access Control")]
            ),
            tags=["prevention", "server-side-enforcement"]
        )

        self.assertEqual(chunk.chunk_id, "owasp-a01-chunk-0")
        self.assertEqual(chunk.document_id, "owasp-top10-a01-2021")
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.total_chunks, 3)
        self.assertEqual(chunk.source, "OWASP Foundation")
        self.assertEqual(chunk.source_type, KnowledgeSourceType.OWASP)
        self.assertEqual(chunk.url, "https://owasp.org/Top10/A01_2021-Broken_Access_Control/")

    def test_chunk_index_bounds_validation(self):
        """Test that negative chunk_index or index exceeding total_chunks raises ValidationError."""
        # Negative chunk_index
        with self.assertRaises(ValidationError):
            KnowledgeChunk(
                chunk_id="chunk-neg",
                document_id="doc-01",
                text="Some text",
                chunk_index=-1,
                source="CWE",
                source_type=KnowledgeSourceType.CWE
            )

        # chunk_index >= total_chunks
        with self.assertRaises(ValidationError):
            KnowledgeChunk(
                chunk_id="chunk-out-of-bounds",
                document_id="doc-01",
                text="Some text",
                chunk_index=5,
                total_chunks=5,  # Valid indices for 5 chunks are 0, 1, 2, 3, 4
                source="CWE",
                source_type=KnowledgeSourceType.CWE
            )

    def test_chunk_validation_rejects_empty_text_and_ids(self):
        """Test that empty or whitespace-only chunk text and IDs raise ValidationError."""
        with self.assertRaises(ValidationError):
            KnowledgeChunk(
                chunk_id="chunk-0",
                document_id="doc-01",
                text="   ",
                chunk_index=0,
                source="NIST",
                source_type=KnowledgeSourceType.NIST
            )

        with self.assertRaises(ValidationError):
            KnowledgeChunk(
                chunk_id="",
                document_id="doc-01",
                text="Valid chunk text",
                chunk_index=0,
                source="NIST",
                source_type=KnowledgeSourceType.NIST
            )

    def test_framework_metadata_representation(self):
        """Test representation of multi-framework metadata."""
        meta = KnowledgeFrameworkMetadata(
            owasp=[OWASPReference(code="A07:2021", name="Identification and Authentication Failures")],
            cwe=[CWEReference(cwe_id="CWE-307", name="Improper Restriction of Excessive Authentication Attempts")],
            mitre=[MITREReference(technique_id="T1110", technique_name="Brute Force", tactic="Credential Access")],
            nist=[NISTReference(control_id="IA-5", title="Authenticator Management", publication="NIST SP 800-53")]
        )

        self.assertEqual(meta.owasp[0].code, "A07:2021")
        self.assertEqual(meta.cwe[0].cwe_id, "CWE-307")
        self.assertEqual(meta.mitre[0].technique_id, "T1110")
        self.assertEqual(meta.nist[0].control_id, "IA-5")

    def test_json_serialization_round_trip(self):
        """Test JSON dump and parse round trip for both KnowledgeDocument and KnowledgeChunk."""
        doc = KnowledgeDocument(
            document_id="cwe-79",
            title="CWE-79: Cross-site Scripting",
            source="MITRE CWE",
            source_type=KnowledgeSourceType.CWE,
            content="The software does not neutralize or incorrectly neutralizes user-controllable input before placing it in output.",
            framework_metadata=KnowledgeFrameworkMetadata(
                cwe=[CWEReference(cwe_id="CWE-79", name="Improper Neutralization of Input During Web Page Generation")]
            )
        )

        doc_json = doc.model_dump_json(indent=2)
        reloaded_doc = KnowledgeDocument.model_validate_json(doc_json)
        self.assertEqual(reloaded_doc.document_id, "cwe-79")
        self.assertEqual(reloaded_doc.framework_metadata.cwe[0].cwe_id, "CWE-79")

        chunk = KnowledgeChunk(
            chunk_id="cwe-79-chunk-0",
            document_id="cwe-79",
            text="Context-aware output encoding is the primary defense against XSS.",
            chunk_index=0,
            total_chunks=1,
            source="MITRE CWE",
            source_type=KnowledgeSourceType.CWE
        )

        chunk_json = chunk.model_dump_json(indent=2)
        reloaded_chunk = KnowledgeChunk.model_validate_json(chunk_json)
        self.assertEqual(reloaded_chunk.chunk_id, "cwe-79-chunk-0")
        self.assertEqual(reloaded_chunk.document_id, "cwe-79")

    def test_extensibility_extra_fields_ignored(self):
        """Ensure extra unexpected fields in raw payloads do not raise ValidationError."""
        raw_doc_payload = {
            "document_id": "doc-ext-01",
            "title": "Extensible Doc",
            "source": "Custom Source",
            "source_type": "CUSTOM",
            "content": "Custom security guidance",
            "future_experimental_field": "some_value"
        }

        doc = KnowledgeDocument.model_validate(raw_doc_payload)
        self.assertEqual(doc.document_id, "doc-ext-01")
        self.assertEqual(doc.source_type, KnowledgeSourceType.CUSTOM)


if __name__ == "__main__":
    unittest.main()
