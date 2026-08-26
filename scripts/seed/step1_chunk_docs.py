"""Backward-compatible shim; implementation lives in rtfm_agent.chunker."""

from rtfm_agent.ingestion.chunker import CHARS_PER_TOKEN, OVERLAP_CHARS, TARGET_CHUNK_CHARS
from rtfm_agent.ingestion.chunker import do_chunk_text, load_and_chunk_docs

__all__ = [
    "CHARS_PER_TOKEN",
    "OVERLAP_CHARS",
    "TARGET_CHUNK_CHARS",
    "do_chunk_text",
    "load_and_chunk_docs",
]
