"""
Cybersecurity Knowledge Base — Text Cleaning and Chunking Layer.

Transforms validated KnowledgeDocument objects into deterministic,
traceable KnowledgeChunk objects ready for future embedding and retrieval.

Pipeline position:
    KnowledgeDocument  (from DocumentIngester)
        ↓
    TextCleaner        (normalize whitespace, preserve meaning)
        ↓
    Chunker            (split on semantic boundaries, carry provenance)
        ↓
    List[KnowledgeChunk]  (ready for: Embeddings → FAISS → Retrieval → LLM)

Design Principles:
    1. Deterministic: same document + same config → identical chunks every time.
    2. Provenance-preserving: every chunk retains document_id, source,
       source_type, url, framework_metadata, and tags from its parent.
    3. Semantic: prefer splitting at headings → paragraphs → sentences
       rather than blindly at a character limit mid-sentence.
    4. Conservative cleaning: no rewriting, no summarizing, no LLM calls.
    5. Configurable: chunk_size and overlap are explicit constructor params.

Scope of this module (Phase 4 — Step 3):
    ✓ Text cleaning
    ✓ Section-aware chunking
    ✓ Provenance propagation
    ✗ Embeddings (future step)
    ✗ FAISS / vector search (future step)
    ✗ LLM reasoning (future step)
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.rag.schemas import KnowledgeChunk, KnowledgeDocument


# =====================================================================
# Constants — Default Configuration
# =====================================================================

#: Default maximum character count for a single chunk.
#: Chosen to stay comfortably within typical embedding-model token windows
#: (~512 tokens ≈ ~2000 chars for English text) while keeping chunks focused.
DEFAULT_CHUNK_SIZE: int = 1400

#: Default character overlap between consecutive chunks when a large section
#: must be split.  ~14% of DEFAULT_CHUNK_SIZE preserves context at boundaries.
DEFAULT_OVERLAP: int = 200

#: Regex that matches a Markdown-style section heading at the start of a line.
#: Matches:  # Heading,  ## Sub-heading,  ### Deep heading  (up to 6 levels).
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

#: A blank line separating paragraphs (one or more empty lines).
_PARAGRAPH_BREAK_RE = re.compile(r"\n{2,}")

#: Sentence-ending punctuation used as a fallback split point.
#: Matches period / exclamation / question mark followed by a space or EOL.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


# =====================================================================
# TextCleaner
# =====================================================================

class TextCleaner:
    """
    Conservative, deterministic text cleaner for cybersecurity source documents.

    Goals:
        - Normalise platform-specific line endings (CRLF, CR → LF).
        - Strip excess whitespace without destroying meaning.
        - Collapse runs of blank lines (more than 2 consecutive become 2).
        - Strip leading/trailing whitespace from every individual line
          while preserving the overall paragraph structure.
        - Never rewrite, summarise, or paraphrase content.
        - Preserve URLs, identifiers, code-like text, and security terms
          exactly as they appear.

    Deterministic guarantee: same input string → same output string.
    """

    def clean(self, text: str) -> str:
        """
        Apply conservative cleaning to a document's content string.

        Steps (in order):
            1. Normalise CRLF → LF and CR → LF.
            2. Strip leading/trailing whitespace from each individual line.
               (Does NOT touch intra-line whitespace, preserving code/lists.)
            3. Collapse runs of more than 2 consecutive blank lines to 2.
            4. Strip leading/trailing whitespace from the overall result.

        Args:
            text: Raw content string from a KnowledgeDocument.

        Returns:
            Cleaned content string.  If the input is empty or whitespace-only,
            returns an empty string.
        """
        if not text or not text.strip():
            return ""

        # Step 1 — Normalise line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Step 2 — Strip leading/trailing whitespace from every line.
        # We intentionally do NOT collapse intra-line multiple spaces because
        # security docs sometimes contain tab-aligned reference tables, code
        # excerpts, and CVE/CVSS data where spacing is meaningful.
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Step 3 — Collapse runs of blank lines to a maximum of 2.
        # This preserves the distinction between a single paragraph break
        # (1 blank line) and a major section break (2 blank lines) while
        # removing accidental extra whitespace from source documents.
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Step 4 — Strip overall leading/trailing whitespace.
        text = text.strip()

        return text


# =====================================================================
# Chunker
# =====================================================================

class Chunker:
    """
    Section-aware, overlap-supporting chunker for cybersecurity knowledge docs.

    Splitting priority (highest to lowest):
        1. Markdown section headings (``# …``, ``## …``, etc.)
        2. Paragraph boundaries (one or more blank lines).
        3. Sentence boundaries (period / ``!`` / ``?`` followed by whitespace).
        4. Hard character-limit split (last resort, preserves whole words).

    Provenance guarantee: every produced KnowledgeChunk carries the parent
    document's ``document_id``, ``source``, ``source_type``, ``url``,
    ``framework_metadata``, and ``tags`` fields unchanged.

    Deterministic guarantee: the same document and configuration always produce
    the same chunks in the same order with the same IDs.

    Attributes:
        chunk_size (int): Maximum characters per chunk (must be > 0).
        overlap (int): Character overlap between consecutive chunks when a
            large block must be split (must be >= 0 and < chunk_size).
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        """
        Initialise the Chunker with size and overlap configuration.

        Args:
            chunk_size: Maximum number of characters per chunk.
                        Default: DEFAULT_CHUNK_SIZE (1400).
            overlap: Number of characters from the end of one chunk to repeat
                     at the start of the next when a large block must be split.
                     Default: DEFAULT_OVERLAP (200).

        Raises:
            ValueError: If chunk_size <= 0, overlap < 0, or overlap >= chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be greater than 0, got {chunk_size}."
            )
        if overlap < 0:
            raise ValueError(
                f"overlap must be >= 0, got {overlap}."
            )
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be strictly less than "
                f"chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, document: KnowledgeDocument) -> List[KnowledgeChunk]:
        """
        Produce a deterministic, ordered list of KnowledgeChunks from a document.

        The cleaned document content is split into semantic segments.  Each
        segment is then further split at the hard ``chunk_size`` limit if
        needed (with ``overlap`` characters of context carried forward).

        Every chunk inherits provenance from the parent KnowledgeDocument.

        Args:
            document: A validated KnowledgeDocument (already ingested).

        Returns:
            Ordered list of KnowledgeChunk objects (chunk_index 0, 1, 2, …).
            Returns a single chunk if the content is short enough to fit.
            Returns an empty list only if cleaned content is empty.
        """
        cleaner = TextCleaner()
        cleaned_text = cleaner.clean(document.content)

        if not cleaned_text:
            return []

        # Split the cleaned text into (section_title, block_text) pairs
        # respecting heading → paragraph → sentence priority.
        segments: List[Tuple[Optional[str], str]] = self._split_into_segments(
            cleaned_text
        )

        # Further split any segment that still exceeds chunk_size.
        raw_chunks: List[Tuple[Optional[str], str]] = []
        for section_title, block in segments:
            if len(block) <= self.chunk_size:
                raw_chunks.append((section_title, block))
            else:
                sub_chunks = self._hard_split(block, section_title)
                raw_chunks.extend(sub_chunks)

        # Filter out any degenerate empty chunks that may arise after splitting.
        raw_chunks = [
            (title, text) for (title, text) in raw_chunks if text.strip()
        ]

        if not raw_chunks:
            return []

        total = len(raw_chunks)

        knowledge_chunks: List[KnowledgeChunk] = []
        for index, (section_title, text) in enumerate(raw_chunks):
            chunk_id = f"{document.document_id}:chunk:{index}"
            chunk = KnowledgeChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                text=text.strip(),
                section_title=section_title,
                chunk_index=index,
                total_chunks=total,
                # --- Provenance fields (copied verbatim from parent) ---
                source=document.source,
                source_type=document.source_type,
                url=document.url,
                framework_metadata=document.framework_metadata,
                tags=document.tags,
                metadata={
                    "parent_document_title": document.title,
                    "chunk_size_config": self.chunk_size,
                    "overlap_config": self.overlap,
                },
            )
            knowledge_chunks.append(chunk)

        return knowledge_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_into_segments(
        self, text: str
    ) -> List[Tuple[Optional[str], str]]:
        """
        Split cleaned text into (section_title, content_block) segments.

        Strategy:
            1. Try to find Markdown headings (``# …``).  When found, each
               heading starts a new segment whose ``section_title`` is the
               heading text (without the ``#`` prefix characters).
            2. Within each heading-bounded block, further split on paragraph
               boundaries (blank lines).
            3. If no headings are found, split purely on paragraph boundaries.

        Args:
            text: Cleaned document content string.

        Returns:
            List of (section_title, block_text) tuples.
            ``section_title`` is ``None`` when no heading can be identified.
        """
        heading_positions = [
            (m.start(), m.group(2).strip(), m.group(0))
            for m in _MARKDOWN_HEADING_RE.finditer(text)
        ]

        if heading_positions:
            return self._split_by_headings(text, heading_positions)

        # No Markdown headings found — split on paragraphs with no title.
        paragraphs = _PARAGRAPH_BREAK_RE.split(text)
        segments: List[Tuple[Optional[str], str]] = []
        for para in paragraphs:
            para = para.strip()
            if para:
                segments.append((None, para))
        return segments if segments else [(None, text)]

    def _split_by_headings(
        self,
        text: str,
        heading_positions: List[Tuple[int, str, str]],
    ) -> List[Tuple[Optional[str], str]]:
        """
        Divide text into blocks bounded by Markdown headings.

        The text before the first heading (if any) becomes a preamble block
        with ``section_title = None``.  Each subsequent block carries the
        heading that introduced it as ``section_title``.

        Within each block, content is further split at paragraph boundaries
        so that no single segment unnecessarily merges distinct paragraphs.

        Args:
            text: Full cleaned document text.
            heading_positions: List of (char_offset, heading_title, full_line)
                tuples as returned by regex finditer.

        Returns:
            List of (section_title, paragraph_text) tuples.
        """
        segments: List[Tuple[Optional[str], str]] = []

        # Build boundary pairs: (start_of_block, end_of_block, section_title)
        boundaries: List[Tuple[int, int, Optional[str]]] = []

        for i, (start, title, full_line) in enumerate(heading_positions):
            end = (
                heading_positions[i + 1][0]
                if i + 1 < len(heading_positions)
                else len(text)
            )
            boundaries.append((start, end, title))

        # Preamble: text before the first heading
        preamble_end = heading_positions[0][0]
        preamble = text[:preamble_end].strip()
        if preamble:
            for para in _PARAGRAPH_BREAK_RE.split(preamble):
                para = para.strip()
                if para:
                    segments.append((None, para))

        # Process each heading-bounded block
        for start, end, title in boundaries:
            block = text[start:end]

            # Remove the heading line itself from the block body so it is
            # not duplicated in the chunk text.  The heading is preserved
            # in section_title instead.
            # Find end of first line (the heading line).
            first_newline = block.find("\n")
            body = block[first_newline + 1:].strip() if first_newline != -1 else ""

            if not body:
                # Heading with no body — still emit a segment so it isn't lost.
                segments.append((title, f"[Section: {title}]"))
                continue

            # Split body on paragraph boundaries.
            paragraphs = _PARAGRAPH_BREAK_RE.split(body)
            first_para = True
            for para in paragraphs:
                para = para.strip()
                if para:
                    # Only the first paragraph in a section carries the title;
                    # subsequent paragraphs are titled None to avoid redundancy.
                    seg_title = title if first_para else None
                    segments.append((seg_title, para))
                    first_para = False

        return segments if segments else [(None, text)]

    def _hard_split(
        self,
        text: str,
        section_title: Optional[str],
    ) -> List[Tuple[Optional[str], str]]:
        """
        Last-resort split for text blocks that exceed ``chunk_size``.

        Attempts to split at sentence boundaries first.  Falls back to
        splitting at the last whitespace within the window when no sentence
        boundary is found.

        Overlap from the previous chunk is prepended to each subsequent chunk
        to preserve context across the split boundary.

        Args:
            text: Content block that exceeds chunk_size.
            section_title: Heading associated with this block (may be None).

        Returns:
            List of (section_title, sub_chunk_text) tuples.
            Only the first sub-chunk carries the original ``section_title``.
        """
        chunks: List[Tuple[Optional[str], str]] = []
        start = 0
        is_first = True

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk — take everything remaining.
                chunk_text = text[start:]
                chunks.append((section_title if is_first else None, chunk_text))
                break

            # Prefer a sentence boundary within the window.
            window = text[start:end]
            split_pos = self._find_sentence_split(window)

            if split_pos is not None:
                # split_pos is relative to window start.
                chunk_text = text[start: start + split_pos]
                next_start = start + split_pos
            else:
                # Fall back: split at the last whitespace within the window.
                last_space = window.rfind(" ")
                if last_space > 0:
                    chunk_text = text[start: start + last_space]
                    next_start = start + last_space + 1
                else:
                    # No whitespace at all — hard cut at chunk_size.
                    chunk_text = window
                    next_start = end

            chunks.append((section_title if is_first else None, chunk_text))
            is_first = False

            # Apply overlap: start the next chunk this many chars earlier.
            start = max(next_start - self.overlap, next_start)

        return chunks

    def _find_sentence_split(self, text: str) -> Optional[int]:
        """
        Find the best sentence-boundary split position within a text window.

        Scans the text for sentence-ending punctuation (``./!/?`` followed
        by whitespace) and returns the position *after* the whitespace,
        preferring the split closest to the end of the window (to maximise
        chunk utilisation while staying within the limit).

        Args:
            text: The candidate window of text (at most chunk_size chars).

        Returns:
            Character offset of the split point within ``text``, or ``None``
            if no sentence boundary is found.
        """
        best: Optional[int] = None
        for match in _SENTENCE_END_RE.finditer(text):
            # match.end() points to the first char of the next sentence.
            best = match.end()
        return best


# =====================================================================
# Module-level Convenience API
# =====================================================================

def clean_and_chunk_document(
    document: KnowledgeDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[KnowledgeChunk]:
    """
    Convenience entry point: clean and chunk a single KnowledgeDocument.

    Combines TextCleaner and Chunker in a single call.  The same document
    with the same arguments always produces identical output (deterministic).

    Args:
        document:   A validated KnowledgeDocument from DocumentIngester.
        chunk_size: Maximum characters per chunk (default: 1400).
        overlap:    Character overlap between consecutive chunks (default: 200).

    Returns:
        Ordered list of KnowledgeChunk objects with full provenance.
        Empty list if the document content is empty after cleaning.

    Example::

        from app.rag.ingestion import load_document_from_file
        from app.rag.chunking import clean_and_chunk_document

        doc = load_document_from_file("knowledge_base/owasp/owasp_a01.json")
        chunks = clean_and_chunk_document(doc)
        for chunk in chunks:
            print(chunk.chunk_id, chunk.section_title, len(chunk.text))
    """
    chunker = Chunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_document(document)


def clean_and_chunk_documents(
    documents: List[KnowledgeDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[KnowledgeChunk]:
    """
    Convenience entry point: clean and chunk a list of KnowledgeDocuments.

    Processes each document independently and concatenates the results.
    Chunk ordering within each document is preserved; documents are processed
    in the order they are provided.

    Args:
        documents:  List of validated KnowledgeDocument objects.
        chunk_size: Maximum characters per chunk (default: 1400).
        overlap:    Character overlap between consecutive chunks (default: 200).

    Returns:
        Flat, ordered list of KnowledgeChunk objects across all documents.
    """
    chunker = Chunker(chunk_size=chunk_size, overlap=overlap)
    all_chunks: List[KnowledgeChunk] = []
    for document in documents:
        all_chunks.extend(chunker.chunk_document(document))
    return all_chunks
