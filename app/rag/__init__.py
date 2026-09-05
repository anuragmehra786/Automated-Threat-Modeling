"""
Cybersecurity Knowledge Base and RAG package.

Exports:
    Schemas (app.rag.schemas):
        - KnowledgeSourceType
        - NISTReference
        - KnowledgeFrameworkMetadata
        - KnowledgeDocument
        - KnowledgeChunk

    Ingestion (app.rag.ingestion):
        - DocumentIngester
        - IngestionError
        - load_document_from_file
        - load_documents_from_directory
        - load_knowledge_documents

    Chunking (app.rag.chunking):
        - TextCleaner
        - Chunker
        - clean_and_chunk_document
        - clean_and_chunk_documents
        - DEFAULT_CHUNK_SIZE
        - DEFAULT_OVERLAP
"""

from app.rag.schemas import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeFrameworkMetadata,
    KnowledgeSourceType,
    NISTReference,
)

from app.rag.ingestion import (
    DocumentIngester,
    IngestionError,
    load_document_from_file,
    load_documents_from_directory,
    load_knowledge_documents,
)

from app.rag.chunking import (
    TextCleaner,
    Chunker,
    clean_and_chunk_document,
    clean_and_chunk_documents,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
)

__all__ = [
    # Schemas
    "KnowledgeSourceType",
    "NISTReference",
    "KnowledgeFrameworkMetadata",
    "KnowledgeDocument",
    "KnowledgeChunk",
    # Ingestion
    "DocumentIngester",
    "IngestionError",
    "load_document_from_file",
    "load_documents_from_directory",
    "load_knowledge_documents",
    # Chunking
    "TextCleaner",
    "Chunker",
    "clean_and_chunk_document",
    "clean_and_chunk_documents",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_OVERLAP",
]
