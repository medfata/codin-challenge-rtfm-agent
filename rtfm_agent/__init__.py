"""RTFM For Me Agent - documentation assistant built on Redis vector search + RAG."""

from rtfm_agent.config import (
    DOCS_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    INDEX_NAME,
    LLM_BASE_URL,
    LLM_MODEL,
    ORT_THREADS,
    REDIS_URL,
    RETRIEVAL_K,
    RETRIEVAL_MAX_DISTANCE,
)

__all__ = [
    "DOCS_DIR",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "GOOGLE_API_KEY",
    "INDEX_NAME",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "ORT_THREADS",
    "REDIS_URL",
    "RETRIEVAL_K",
    "RETRIEVAL_MAX_DISTANCE",
]
