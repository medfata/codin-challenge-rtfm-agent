"""Corpus ingestion: document loading, chunking, embedding, Redis indexing,
and content-hash version tracking."""

from rtfm_agent.ingestion import documents, versioning
from rtfm_agent.ingestion.chunker import do_chunk_text, load_and_chunk_docs
from rtfm_agent.ingestion.pipeline import (
    collect_docs,
    create_redis_index,
    delete_source_keys,
    run_ingestion,
    store_in_redis,
)

__all__ = [
    "collect_docs",
    "create_redis_index",
    "delete_source_keys",
    "documents",
    "do_chunk_text",
    "load_and_chunk_docs",
    "run_ingestion",
    "store_in_redis",
    "versioning",
]
