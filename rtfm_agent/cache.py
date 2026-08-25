"""Semantic cache: serve answers for repeated/similar questions without an LLM call.

Entries live under the tenant-scoped `t:{org}:cache:` prefix as HASHes:
  question  TEXT     - original question (searchable, informational)
  response  TEXT     - cached LLM answer
  citations TEXT     - JSON array of citation dicts
  embedding VECTOR   - question embedding (FLAT / COSINE)

Each tenant has its own FT index ({prefix}cache_idx) covering only its keys.
Every operation is fail-open with logging: a broken cache must never take
the assistant down - we simply fall through to the full RAG pipeline.
"""

import json
import logging
import uuid

import numpy as np
from redis import Redis

from rtfm_agent.config import CACHE_THRESHOLD, EMBEDDING_DIM
from rtfm_agent.tenancy import TenantContext

logger = logging.getLogger(__name__)


def ensure_cache_index(r: Redis, t: TenantContext, force: bool = False) -> bool:
    """Create the tenant's FT index if missing or wrong dim; True if created."""
    try:
        info = r.execute_command("FT.INFO", t.cache_index)

        def find_dim(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(k, bytes) and k.lower() == b"dim" and isinstance(v, (int, float)):
                        return int(v)
                    found = find_dim(v)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = find_dim(item)
                    if found is not None:
                        return found
            return None

        if not force and find_dim(info) == EMBEDDING_DIM:
            return False
    except Exception:
        pass

    try:
        r.execute_command("FT.DROPINDEX", t.cache_index)
    except Exception:
        pass

    r.execute_command(
        "FT.CREATE", t.cache_index,
        "ON", "HASH",
        "PREFIX", "1", f"{t.prefix}cache:",
        "SCHEMA",
        "question", "TEXT",
        "embedding", "VECTOR", "FLAT", "6",
            "TYPE", "FLOAT32",
            "DIM", str(EMBEDDING_DIM),
            "DISTANCE_METRIC", "COSINE",
    )
    return True


def _normalize(obj):
    """Recursively decode redis-py's bytes-keyed FT.SEARCH response."""
    if isinstance(obj, bytes):
        return obj.decode(errors="replace")
    if isinstance(obj, dict):
        return {_normalize(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    return obj


def lookup(r: Redis, t: TenantContext, qvec: bytes) -> dict | None:
    """Return {question, response, citations, score, corpus_version} on a hit.

    `corpus_version` is the version whose docs produced the answer (0 when
    the entry predates version tracking) - the caller compares it against
    the live corpus to warn about stale answers.

    Searches the per-tenant index only. Self-healing: a missing/corrupt
    tenant index is recreated once and retried.
    """
    query = "*=>[KNN 1 @embedding $vec AS score]"
    for attempt in (0, 1):
        try:
            res = r.execute_command(
                "FT.SEARCH", t.cache_index,
                query,
                "PARAMS", "2", "vec", qvec,
                "SORTBY", "score",
                "DIALECT", "2",
                "RETURN", "5", "question", "response", "citations",
                "corpus_version", "score",
            )
            break
        except Exception as exc:
            logger.warning("cache lookup failed (attempt %d): %s", attempt + 1, exc)
            if attempt == 0:
                try:
                    ensure_cache_index(r, t, force=True)
                    continue
                except Exception as exc2:
                    logger.warning("cache index recreation failed: %s", exc2)
            return None

    data = _normalize(res)
    docs = data.get("results") or [] if isinstance(data, dict) else []
    if not docs:
        return None

    fields = docs[0].get("extra_attributes") or docs[0].get("fields") or {}
    try:
        score = float(fields.get("score", 99.0))
    except (TypeError, ValueError):
        return None
    if score > CACHE_THRESHOLD:
        return None

    try:
        citations = json.loads(fields.get("citations") or "[]")
    except json.JSONDecodeError:
        citations = []

    try:
        corpus_version = int(fields.get("corpus_version") or 0)
    except (TypeError, ValueError):
        corpus_version = 0

    return {
        "question": fields.get("question", ""),
        "response": fields.get("response", ""),
        "citations": citations,
        "score": score,
        "corpus_version": corpus_version,
    }


def store(r: Redis, t: TenantContext, question: str, qvec: bytes,
          response: str, citations: list, corpus_version: int = 0) -> str:
    """Persist a question/answer pair for future semantic hits (tenant-scoped).

    `corpus_version` records which doc version produced the answer so later
    hits can be flagged when the corpus has moved on.
    """
    try:
        key = f"{t.prefix}cache:{uuid.uuid4().hex}"
        emb = qvec if isinstance(qvec, (bytes, bytearray)) else np.asarray(qvec, dtype=np.float32).tobytes()
        r.hset(key, mapping={
            "question": question,
            "response": response,
            "citations": json.dumps(citations, ensure_ascii=False),
            "corpus_version": str(corpus_version),
            "embedding": emb,
        })
        return key
    except Exception as exc:
        logger.warning("cache store failed (non-fatal): %s", exc)
        return ""


def flush(r: Redis, t: TenantContext) -> int:
    """Delete all of the tenant's cache entries; returns how many were removed."""
    cursor = 0
    removed = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=f"{t.prefix}cache:*", count=500)
        if keys:
            r.delete(*keys)
            removed += len(keys)
        if cursor == 0:
            break
    try:
        ensure_cache_index(r, t, force=True)
    except Exception as exc:
        logger.warning("cache index recreation after flush failed: %s", exc)
    return removed


def count_entries(r: Redis, t: TenantContext) -> int:
    """Number of cached question/answer pairs for this tenant."""
    total = 0
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=f"{t.prefix}cache:*", count=500)
        total += len(keys)
        if cursor == 0:
            break
    return total
