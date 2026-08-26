"""Chunk documentation text into ~500 token segments with paragraph overlap."""

import re
from typing import Generator


# Approximate char-to-token ratio for English text (~4 chars per token)
CHARS_PER_TOKEN = 4
TARGET_CHUNK_CHARS = 500 * CHARS_PER_TOKEN  # ~2000 chars
OVERLAP_CHARS = 50 * CHARS_PER_TOKEN  # ~200 chars overlap


def do_chunk_text(content: str, target_chars: int = TARGET_CHUNK_CHARS, overlap_chars: int = OVERLAP_CHARS) -> Generator[tuple[int, str], None, None]:
    """Split content into chunks with paragraph-aware boundaries and overlap.

    Yields (chunk_pos, chunk_text) tuples where chunk_pos is 1-indexed.
    - Splits on double-newlines (paragraph boundaries)
    - Accumulates paragraphs until target_chars is reached
    - Carries overlap_chars from end of previous chunk into next chunk start
    - Respects paragraph boundaries (avoids cutting mid-sentence)
    """
    # Split on double-newlines to get paragraphs
    paragraphs = re.split(r"\n\n+", content.strip())

    if not paragraphs or all(p.strip() == "" for p in paragraphs):
        return

    # Track accumulated text and position
    current_chunk_chars = 0
    current_paragraphs: list[str] = []
    chunk_pos = 0

    # Track the tail of the last yielded chunk for overlap
    last_tail: str = ""

    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            continue

        para_chars = len(para_stripped)

        # If single paragraph exceeds target, split it into sub-chunks by sentence
        if para_chars > target_chars:
            # Yield what we have so far, applying overlap from previous
            if current_paragraphs:
                chunk_pos += 1
                chunk_str = " ".join(current_paragraphs).strip()
                if chunk_str:
                    final_text = _apply_overlap_prev(chunk_str, last_tail)
                    yield chunk_pos, final_text
                    # Update last_tail: save tail of this chunk
                    last_tail = chunk_str[-overlap_chars:] if chunk_str else ""
                current_paragraphs = []
                current_chunk_chars = 0

            # Split the oversized paragraph into sub-chunks by sentence
            sentences = re.split(r"(?<=[.!?])\s+", para_stripped)
            sub_chunk_chars = 0
            sub_parts: list[str] = []
            for sent in sentences:
                sent_len = len(sent)
                if sub_chunk_chars + sent_len > target_chars and sub_parts:
                    chunk_pos += 1
                    sub_text = " ".join(sub_parts).strip()
                    if sub_text:
                        final_text = _apply_overlap_prev(sub_text, last_tail)
                        yield chunk_pos, final_text
                        last_tail = sub_text[-overlap_chars:] if sub_text else ""
                    sub_parts = [sent]
                    sub_chunk_chars = sent_len
                else:
                    sub_parts.append(sent)
                    sub_chunk_chars += sent_len
            if sub_parts:
                chunk_pos += 1
                sub_text = " ".join(sub_parts).strip()
                if sub_text:
                    final_text = _apply_overlap_prev(sub_text, last_tail)
                    yield chunk_pos, final_text
                    last_tail = sub_text[-overlap_chars:] if sub_text else ""
            continue

        # Would adding this paragraph exceed the target?
        if current_chunk_chars + para_chars > target_chars and current_paragraphs:
            # Yield current chunk, applying overlap from previous
            chunk_pos += 1
            chunk_str = " ".join(current_paragraphs).strip()
            if chunk_str:
                final_text = _apply_overlap_prev(chunk_str, last_tail)
                yield chunk_pos, final_text
                last_tail = chunk_str[-overlap_chars:] if chunk_str else ""
            else:
                last_tail = ""

            # Start new chunk with overlap carry from previous chunk's tail
            # The overlap text becomes the first paragraph of the new chunk
            current_paragraphs = [last_tail] if last_tail else []
            current_chunk_chars = len(last_tail) if last_tail else 0
            last_tail = ""

        # Add paragraph to current chunk
        current_paragraphs.append(para_stripped)
        current_chunk_chars += para_chars

    # Yield last chunk, apply overlap from previous
    if current_paragraphs:
        chunk_pos += 1
        chunk_str = " ".join(current_paragraphs).strip()
        if chunk_str:
            final_text = _apply_overlap_prev(chunk_str, last_tail)
            yield chunk_pos, final_text


def _apply_overlap_prev(chunk_str: str, prev_tail: str) -> str:
    """Apply overlap by prepending previous chunk's trailing text.

    If prev_tail is empty, return chunk_str unchanged.
    Otherwise, prepend a sensible overlap portion (~50 chars).
    """
    if not prev_tail:
        return chunk_str

    # Take last portion for overlap (up to 50 chars)
    overlap_portion = prev_tail[-50:] if len(prev_tail) > 50 else prev_tail
    if overlap_portion and chunk_str:
        return f"{overlap_portion} {chunk_str}"
    return chunk_str


def load_and_chunk_docs(docs, max_chunks_per_doc=None) -> list:
    """Chunk all documents and return list of {source_file, heading, chunk_pos, chunk_text}.

    Args:
        docs: List of dicts from ingestion.documents.load_asc_files()
        max_chunks_per_doc: Optional limit for testing (None = all chunks)

    Returns:
        List of dicts with keys: source_file, heading, chunk_pos, chunk_text
    """
    all_chunks: list = []

    for doc in docs:
        source_file = doc["source_file"]
        heading = doc["heading"]
        content = doc["content"]

        chunks = list(do_chunk_text(content))

        # Apply max_chunks_per_doc limit if specified
        if max_chunks_per_doc is not None:
            chunks = chunks[:max_chunks_per_doc]

        for chunk_pos, chunk_text in chunks:
            all_chunks.append({
                "source_file": source_file,
                "heading": heading,
                "chunk_pos": chunk_pos,
                "chunk_text": chunk_text,
            })

    return all_chunks
