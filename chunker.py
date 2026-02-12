"""
Smart Text Chunking Module.
Splits extracted text into overlapping chunks that respect
paragraph and sentence boundaries for better retrieval quality.
"""

import re
import logging
from typing import Optional

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class RecursiveChunker:
    """
    Recursively splits text into chunks using a hierarchy of separators.
    Tries to keep semantically meaningful units (paragraphs, sentences) intact.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        """
        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    def chunk_text(
        self,
        text: str,
        source: str = "unknown",
    ) -> list[dict]:
        """
        Split text into overlapping chunks with metadata.

        Args:
            text: The full text to split.
            source: Source filename for metadata.

        Returns:
            List of dicts with keys: 'text', 'source', 'chunk_index'.
        """
        if not text or not text.strip():
            return []

        text = self._clean_text(text)
        raw_chunks = self._recursive_split(text, 0)

        # Apply overlap
        chunks_with_overlap = self._apply_overlap(raw_chunks)

        # Build result with metadata
        result = []
        for i, chunk in enumerate(chunks_with_overlap):
            if chunk.strip():
                result.append({
                    "text": chunk.strip(),
                    "source": source,
                    "chunk_index": i,
                })

        logger.info(
            f"Chunked '{source}': {len(text)} chars → {len(result)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})."
        )
        return result

    # ─── Private Methods ─────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and remove excessive blank lines."""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _recursive_split(self, text: str, sep_index: int) -> list[str]:
        """
        Recursively split text using separators in priority order.
        Falls back to character-level splitting as last resort.
        """
        if len(text) <= self.chunk_size:
            return [text]

        if sep_index >= len(self.separators):
            # Last resort: hard split by chunk_size
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        separator = self.separators[sep_index]
        parts = text.split(separator)

        chunks = []
        current_chunk = ""

        for part in parts:
            candidate = (
                current_chunk + separator + part if current_chunk else part
            )

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if len(part) > self.chunk_size:
                    # Part itself is too large, split recursively
                    sub_chunks = self._recursive_split(part, sep_index + 1)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlapping context from the previous chunk's tail."""
        if not chunks or self.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            overlap = chunks[i - 1][-self.chunk_overlap :]
            result.append(overlap + chunks[i])

        return result
