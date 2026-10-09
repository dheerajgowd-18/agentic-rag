import re
from typing import List
import logfire


def _split_by_characters(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Fallback when text cannot be split further by delimiters without exceeding chunk_size."""
    if not text.strip():
        return []
    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for i in range(0, len(text), step):
        piece = text[i : i + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if i + chunk_size >= len(text):
            break
    return chunks


def _split_text_recursive(
    text: str,
    delimiters: List[str],
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Recursively splits text using a hierarchy of semantic delimiters.
    Guarantees chunks do not exceed chunk_size and preserves chunk_overlap across boundaries.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    if len(clean_text) <= chunk_size:
        return [clean_text]

    if not delimiters:
        return _split_by_characters(clean_text, chunk_size, chunk_overlap)

    delimiter = delimiters[0]
    sub_delimiters = delimiters[1:]

    # Split using the current delimiter
    parts = clean_text.split(delimiter)
    chunks: List[str] = []
    current_chunk = ""

    for i, part in enumerate(parts):
        if not part.strip():
            continue

        piece = part if not current_chunk else delimiter + part

        if len(current_chunk) + len(piece) <= chunk_size:
            current_chunk += piece
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                # Compute overlap tail
                overlap_len = min(len(current_chunk), chunk_overlap)
                tail = current_chunk[-overlap_len:] if overlap_len > 0 else ""
                current_chunk = (tail + delimiter + part) if tail else part
            else:
                # Single part is larger than chunk_size, recurse with finer delimiters
                sub_chunks = _split_text_recursive(part, sub_delimiters, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
                current_chunk = ""

            # Check if current_chunk with tail already exceeds chunk_size
            if len(current_chunk) > chunk_size:
                sub_chunks = _split_text_recursive(current_chunk, sub_delimiters, chunk_size, chunk_overlap)
                if sub_chunks:
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1]
                else:
                    current_chunk = ""

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c.strip()]


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """
    Sentence- and paragraph-aware recursive text chunker.
    Splits along paragraphs -> newlines -> sentences -> clauses -> words -> characters.
    Enforces chunk_size and chunk_overlap invariants strictly.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text) if text else 0):
        if not text or not text.strip():
            return []

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")

        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 4)
            logfire.warning(f"chunk_overlap adjusted to {chunk_overlap} (must be less than chunk_size {chunk_size})")

        delimiters = ["\n\n", "\n", ". ", "; ", ", ", " "]
        valid_chunks = _split_text_recursive(text, delimiters, chunk_size, chunk_overlap)
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks with max size {chunk_size} and {chunk_overlap} overlap")
        return valid_chunks
