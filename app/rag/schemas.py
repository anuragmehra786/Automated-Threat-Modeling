"""
Pydantic Data Contracts for Cybersecurity Knowledge Base and RAG Pipeline.

Defines schemas for:
1. KnowledgeSourceType: Authoritative cybersecurity knowledge sources (OWASP, CWE, MITRE, NIST, etc.)
2. NISTReference & KnowledgeFrameworkMetadata: Structured framework metadata schemas.
3. KnowledgeDocument: Original authoritative cybersecurity reference documents.
4. KnowledgeChunk: Traceable knowledge chunks extracted for semantic indexing and retrieval.

Key Principles:
- Strict Provenance & Traceability: Every chunk preserves document_id, source, source_type, and URL.
- Framework Metadata Preservation: Supports OWASP, CWE, MITRE ATT&CK, and NIST references.
- Separation of Concerns: External security knowledge is strictly decoupled from target-application
  epistemic statuses (OBSERVED/INFERRED/UNKNOWN).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.schemas import CWEReference, MITREReference, OWASPReference


class KnowledgeSourceType(str, Enum):
    """
    Classification of cybersecurity knowledge document sources.
    """
    OWASP = "OWASP"
    CWE = "CWE"
    MITRE_ATTACK = "MITRE_ATTACK"
    NIST = "NIST"
    CAPEC = "CAPEC"
    GENERAL_SECURITY = "GENERAL_SECURITY"
    CUSTOM = "CUSTOM"


class NISTReference(BaseModel):
    """Reference to a NIST Special Publication or control framework (e.g. NIST SP 800-53, CSF)."""
    model_config = ConfigDict(extra="ignore")

    control_id: str = Field(..., description="NIST control identifier (e.g., 'AC-2', 'IA-5', 'SC-8')", examples=["AC-2"])
    title: str = Field(..., description="Control title or specification name", examples=["Account Management"])
    publication: str = Field(default="NIST SP 800-53", description="Specific NIST publication or framework standard", examples=["NIST SP 800-53"])
    description: Optional[str] = Field(None, description="Summary of the control requirement")
    url: Optional[str] = Field(None, description="Authoritative reference URL")


class KnowledgeFrameworkMetadata(BaseModel):
    """
    Container for cybersecurity framework references attached to knowledge documents and chunks.
    """
    model_config = ConfigDict(extra="ignore")

    owasp: List[OWASPReference] = Field(default_factory=list, description="Associated OWASP references")
    cwe: List[CWEReference] = Field(default_factory=list, description="Associated CWE references")
    mitre: List[MITREReference] = Field(default_factory=list, description="Associated MITRE ATT&CK references")
    nist: List[NISTReference] = Field(default_factory=list, description="Associated NIST control references")


class KnowledgeDocument(BaseModel):
    """
    Represents an original authoritative cybersecurity reference document.
    
    Serves as the root knowledge item ingested into the security knowledge base.
    """
    model_config = ConfigDict(extra="ignore")

    document_id: str = Field(..., description="Unique identifier for the knowledge document", examples=["owasp-top10-a01-2021"])
    title: str = Field(..., description="Document title", examples=["OWASP Top 10:2021 - A01 Broken Access Control"])
    source: str = Field(..., description="Authoritative source organization or standard", examples=["OWASP Foundation"])
    source_type: KnowledgeSourceType = Field(..., description="Classification category of the knowledge source", examples=[KnowledgeSourceType.OWASP])
    url: Optional[str] = Field(None, description="Canonical URL to the authoritative publication")
    version: Optional[str] = Field(None, description="Standard or document version string", examples=["2021"])
    publication_date: Optional[str] = Field(None, description="Publication or release date string", examples=["2021-09-24"])
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of document retrieval/ingestion")
    content: str = Field(..., description="Full text content of the security knowledge document")
    framework_metadata: KnowledgeFrameworkMetadata = Field(default_factory=KnowledgeFrameworkMetadata, description="Associated cybersecurity framework mappings")
    tags: List[str] = Field(default_factory=list, description="Descriptive classification tags for indexing")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible auxiliary metadata")

    @field_validator("document_id", "title", "source", mode="after")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace-only.")
        return v.strip()

    @field_validator("content", mode="after")
    @classmethod
    def validate_content_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Document content cannot be empty or whitespace-only.")
        return v


class KnowledgeChunk(BaseModel):
    """
    Represents a discrete, traceable text chunk extracted from a KnowledgeDocument.
    
    Preserves document-level provenance (document_id, source, source_type, URL)
    to guarantee traceability when retrieved during RAG workflows.
    """
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(..., description="Unique identifier for this knowledge chunk", examples=["owasp-top10-a01-2021-chunk-0"])
    document_id: str = Field(..., description="Identifier of the parent KnowledgeDocument for provenance tracking", examples=["owasp-top10-a01-2021"])
    text: str = Field(..., description="Text content of the chunk extracted for semantic indexing")
    section_title: Optional[str] = Field(None, description="Section or heading title within the parent document")
    chunk_index: int = Field(..., ge=0, description="0-indexed position of this chunk within the parent document")
    total_chunks: Optional[int] = Field(None, ge=1, description="Total number of chunks produced from the parent document")
    source: str = Field(..., description="Inherited source organization or standard from parent document")
    source_type: KnowledgeSourceType = Field(..., description="Inherited source classification from parent document")
    url: Optional[str] = Field(None, description="Inherited canonical URL from parent document")
    framework_metadata: KnowledgeFrameworkMetadata = Field(default_factory=KnowledgeFrameworkMetadata, description="Inherited or section-specific framework mappings")
    tags: List[str] = Field(default_factory=list, description="Inherited or chunk-specific classification tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible chunk-level metadata")

    @field_validator("chunk_id", "document_id", mode="after")
    @classmethod
    def validate_ids_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ID fields cannot be empty or whitespace-only.")
        return v.strip()

    @field_validator("text", mode="after")
    @classmethod
    def validate_text_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk text cannot be empty or whitespace-only.")
        return v

    @model_validator(mode="after")
    def validate_chunk_bounds(self) -> KnowledgeChunk:
        if self.total_chunks is not None and self.chunk_index >= self.total_chunks:
            raise ValueError(f"chunk_index ({self.chunk_index}) must be less than total_chunks ({self.total_chunks}).")
        return self
