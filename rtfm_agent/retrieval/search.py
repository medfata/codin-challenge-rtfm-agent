"""Question embedding + KNN retrieval over the tenant's Redis docs index."""

import logging

from redis import Redis

from rtfm_agent.common.redis_utils import escape_tag, normalize
from rtfm_agent.common.tenancy import TenantContext
from rtfm_agent.config import settings

logger = logging.getLogger(__name__)


def retrieve(question: str, r: Redis, t: TenantContext, k: int | None = None,
             qvec: bytes | None = None, doc_filter: set[str] | None = None) -> list[dict]:
    """Return the top-k relevant chunks for a question.

    Chunks with cosine distance above retrieval.max_distance are dropped,
    so unrelated questions yield an empty list instead of noise.
    Pass `qvec` (float32 blob) to reuse an embedding computed upstream.
    Pass `doc_filter` (set of source_file values) for hybrid search: the
    vector KNN runs only over matching documents.

    Searches the tenant's own index (`t.doc_index`).
    """
    k = k or settings.retrieval.k
    if qvec is None:
        from rtfm_agent.embedder import embed_question

        qvec = embed_question(question)

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

    data = normalize(res)
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
        if score > settings.retrieval.max_distance:
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

    data = normalize(res)
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
