"""
Cybersecurity Knowledge Base Ingestion Layer.

Loads and validates local cybersecurity source documents (e.g., OWASP, CWE, MITRE, NIST)
into structured, typed KnowledgeDocument objects.

Key Principles:
1. Strict Validation: Every source file is validated against the KnowledgeDocument Pydantic contract.
2. Provenance Preservation: Retains document IDs, source organizations, source types, and URLs.
3. Determinism: The same file or directory produces identical document content and identity.
4. Error Handling: Descriptive IngestionError exceptions with actionable diagnostics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Union
from pydantic import ValidationError

from app.rag.schemas import KnowledgeDocument


class IngestionError(Exception):
    """Base exception for knowledge base ingestion errors."""
    pass


class DocumentIngester:
    """
    Ingests and validates local cybersecurity reference documents from files and directories.
    """

    def load_document(self, file_path: Union[str, Path]) -> KnowledgeDocument:
        """
        Loads and validates a single KnowledgeDocument from a local JSON file.
        
        Args:
            file_path: Path to the JSON knowledge document.
            
        Returns:
            Validated KnowledgeDocument instance.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            IngestionError: If the file is not a valid JSON document or fails schema validation.
        """
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Knowledge document file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise IngestionError(f"Malformed JSON in knowledge document '{path.name}': {e}") from e
        except Exception as e:
            raise IngestionError(f"Failed to read knowledge document '{path.name}': {e}") from e

        if not isinstance(raw_data, dict):
            raise IngestionError(f"Knowledge document '{path.name}' must contain a JSON object, got {type(raw_data).__name__}.")

        try:
            document = KnowledgeDocument.model_validate(raw_data)
            return document
        except ValidationError as e:
            raise IngestionError(f"Schema validation failed for knowledge document '{path.name}': {e}") from e

    def load_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> List[KnowledgeDocument]:
        """
        Loads and validates all JSON knowledge documents from a directory.
        
        Args:
            directory_path: Path to the directory containing knowledge files.
            recursive: Whether to search recursively in subdirectories (default: True).
            
        Returns:
            Sorted list of validated KnowledgeDocument objects (sorted deterministically by document_id).
            
        Raises:
            FileNotFoundError: If the specified directory does not exist.
            IngestionError: If any document fails parsing or validation.
        """
        dir_path = Path(directory_path).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Knowledge base directory not found: {dir_path}")

        pattern = "**/*.json" if recursive else "*.json"
        json_files = sorted(dir_path.glob(pattern))

        documents: List[KnowledgeDocument] = []
        for file_path in json_files:
            # Skip hidden files or lock files
            if file_path.name.startswith("."):
                continue
            doc = self.load_document(file_path)
            documents.append(doc)

        # Sort deterministically by document_id
        documents.sort(key=lambda d: d.document_id)
        return documents


# =====================================================================
# Module-level Convenience API Functions
# =====================================================================

def load_document_from_file(file_path: Union[str, Path]) -> KnowledgeDocument:
    """
    Convenience function to load and validate a single knowledge document.
    """
    ingester = DocumentIngester()
    return ingester.load_document(file_path)


def load_documents_from_directory(directory_path: Union[str, Path], recursive: bool = True) -> List[KnowledgeDocument]:
    """
    Convenience function to load and validate all knowledge documents in a directory.
    """
    ingester = DocumentIngester()
    return ingester.load_directory(directory_path, recursive=recursive)


def load_knowledge_documents(source_path: Union[str, Path], recursive: bool = True) -> List[KnowledgeDocument]:
    """
    Unified entry point: loads knowledge documents from a single file or a directory.
    """
    path = Path(source_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Knowledge source path not found: {path}")

    ingester = DocumentIngester()
    if path.is_file():
        return [ingester.load_document(path)]
    return ingester.load_directory(path, recursive=recursive)
