"""Question embedding + KNN retrieval over the Redis docs index."""

import logging

import numpy as np
from redis import Redis

from rtfm_agent.config import RETRIEVAL_K, RETRIEVAL_MAX_DISTANCE
from rtfm_agent.embedder import FastInt8Embedder
from rtfm_agent.tenancy import TenantContext

logger = logging.getLogger(__name__)

_embedder: FastInt8Embedder | None = None


def get_embedder() -> FastInt8Embedder:
    global _embedder
    if _embedder is None:
        from rtfm_agent.config import ORT_THREADS

        _embedder = FastInt8Embedder(threads=ORT_THREADS)
    return _embedder


def _normalize(obj):
    """Recursively decode redis-py's bytes-keyed FT.SEARCH response."""
    if isinstance(obj, bytes):
        return obj.decode(errors="replace")
    if isinstance(obj, dict):
        return {_normalize(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    return obj


def escape_tag(value: str) -> str:
    """Escape punctuation for use inside a RediSearch TAG query set."""
    out = []
    for ch in value:
        if ch.isalnum():
            out.append(ch)
        elif ch == ",":
            continue  # comma is the OR separator inside {...}
        else:
            out.append("\\" + ch)
    return "".join(out)


def retrieve(question: str, r: Redis, t: TenantContext, k: int | None = None,
             qvec: bytes | None = None, doc_filter: set[str] | None = None) -> list[dict]:
    """Return the top-k relevant chunks for a question.

    Chunks with cosine distance above RETRIEVAL_MAX_DISTANCE are dropped,
    so unrelated questions yield an empty list instead of noise.
    Pass `qvec` (float32 blob) to reuse an embedding computed upstream.
    Pass `doc_filter` (set of source_file values) for hybrid search: the
    vector KNN runs only over matching documents.

    Searches the tenant's own index (`t.doc_index`).
    """
    k = k or RETRIEVAL_K
    if qvec is None:
        qvec = get_embedder().embed([question])[0].astype(np.float32).tobytes()

    knn = f"[KNN {k} @embedding $vec AS score]"
    if doc_filter:
        # NOTE: comma-separated OR sets inside {} return 0 results in hybrid
        # pre-filters on this RediSearch build - pipe union works reliably.
        tag_set = "|".join(escape_tag(f) for f in sorted(doc_filter))
        query = f"(@source_file:{{{tag_set}}})=>{knn}"
    else:
        query = f"*=>{knn}"

    res = r.execute_command(
        "FT.SEARCH", t.doc_index,
        query,
        "PARAMS", "2", "vec", qvec,
        "SORTBY", "score",
        "DIALECT", "2",
        "RETURN", "5", "text", "source_file", "section_heading", "chunk_pos", "score",
    )

    data = _normalize(res)
    if isinstance(data, dict):
        docs = data.get("results") or []
    else:
        # legacy flat reply: [total, key1, fields1, key2, fields2, ...]
        docs = []
        for i in range(1, len(data), 2):
            flat = data[i + 1]
            attrs = {
                flat[j]: flat[j + 1] for j in range(0, len(flat), 2)
            } if isinstance(flat, list) else {}
            docs.append({"extra_attributes": attrs})

    chunks: list[dict] = []
    for doc in docs:
        fields = doc.get("extra_attributes") or doc.get("fields") or {}
        try:
            score = float(fields.get("score", doc.get("score", 99.0)))
        except (TypeError, ValueError):
            continue
        if score > RETRIEVAL_MAX_DISTANCE:
            continue
        chunks.append({
            "source_file": fields["source_file"],
            "section_heading": fields.get("section_heading", ""),
            "chunk_pos": int(fields.get("chunk_pos", 0)),
            "text": fields["text"],
            "score": score,
        })
    return chunks


def indexed_sources(r: Redis, t: TenantContext) -> list[dict]:
    """Distinct source_file values with chunk counts (FT.AGGREGATE GROUPBY).

    Aggregates over the tenant's own index (`t.doc_index`).
    """
    try:
        res = r.execute_command(
            "FT.AGGREGATE", t.doc_index,
            "*",
            "GROUPBY", "1", "@source_file",
            "REDUCE", "COUNT", "0", "AS", "chunks",
            "SORTBY", "2", "@source_file", "ASC",
            "LIMIT", "0", "500",
        )
    except Exception as exc:
        logger.warning("indexed_sources aggregation failed: %s", exc)
        return []

    data = _normalize(res)
    rows = data[1:] if isinstance(data, list) else (data.get("results") or [])
    out: list[dict] = []
    for row in rows:
        if isinstance(row, dict):
            flat = row.get("extra_attributes") or row.get("fields") or row
        else:
            flat = row
        fields: dict = {}
        if isinstance(flat, list):
            for i in range(0, len(flat) - 1, 2):
                fields[flat[i]] = flat[i + 1]
        elif isinstance(flat, dict):
            fields = flat
        src = fields.get("source_file")
        if not src:
            continue
        try:
            chunks = int(fields.get("chunks", 0))
        except (TypeError, ValueError):
            chunks = 0
        out.append({"source_file": src, "chunks": chunks})
    return out
