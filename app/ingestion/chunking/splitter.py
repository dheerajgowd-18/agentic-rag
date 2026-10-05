from typing import List
import re
import logfire


def _split_text_with_delimiters(text: str, delimiters: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Recursively splits text using a hierarchy of delimiters, ensuring chunks
    stay under chunk_size while preserving chunk_overlap characters across boundaries.
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    if not delimiters:
        # Fallback character slice with overlap
        chunks = []
        step = max(1, chunk_size - chunk_overlap)
        for i in range(0, len(text), step):
            piece = text[i : i + chunk_size].strip()
            if piece:
                chunks.append(piece)
        return chunks

    delimiter = delimiters[0]
    sub_delimiters = delimiters[1:]

    splits = text.split(delimiter)
    chunks: List[str] = []
    current_chunk = ""

    for s in splits:
        piece = s if not current_chunk else delimiter + s
        if len(current_chunk) + len(piece) <= chunk_size:
            current_chunk += piece
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                # Keep sliding overlap tail from the previous chunk
                overlap_len = min(len(current_chunk), chunk_overlap)
                tail = current_chunk[-overlap_len:] if overlap_len > 0 else ""
                current_chunk = (tail + delimiter + s) if tail else s
            else:
                # If a single piece exceeds chunk_size, split with finer delimiters
                sub_chunks = _split_text_with_delimiters(s, sub_delimiters, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
                current_chunk = ""

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c.strip()]


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """
    Recursive sentence-aware chunker.
    Splits along paragraphs -> newlines -> sentences -> words, maintaining
    a sliding overlap between adjacent chunks to preserve semantic context.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text or not text.strip():
            return []

        delimiters = ["\n\n", "\n", ". ", "; ", " ", ""]
        valid_chunks = _split_text_with_delimiters(text, delimiters, chunk_size, chunk_overlap)
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks with {chunk_overlap} char overlap")
        return valid_chunks

