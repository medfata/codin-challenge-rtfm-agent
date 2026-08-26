"""Document ingestion pipeline: load -> chunk -> embed -> index in Redis."""

import time
from pathlib import Path

import numpy as np
from redis import Redis

from rtfm_agent.chunker import load_and_chunk_docs
from rtfm_agent.config import DOCS_DIR, EMBEDDING_DIM
from rtfm_agent.documents import load_asc_files
from rtfm_agent.embedder import FastInt8Embedder
from rtfm_agent.tenancy import TenantContext
from rtfm_agent import events as events_mod
from rtfm_agent import versions as versions_mod


def collect_docs(docs_dir: str | Path, t: TenantContext) -> list[dict]:
    """Primary corpus plus this tenant's approved web-crawl pages.

    Web pages live in their own root (docs/web/<org>/<host>/<page>.md) and
    are namespaced `web/<host>/<page>.md` so they can never collide with
    local files and citations visibly mark web-sourced chunks.
    """
    docs = load_asc_files(str(docs_dir), recursive=True)
    web_root = versions_mod.web_docs_root(t)
    if web_root.is_dir():
        for doc in load_asc_files(str(web_root), recursive=True):
            docs.append({**doc, "source_file": f"web/{doc['source_file']}"})
    return docs


def delete_source_keys(r: Redis, t: TenantContext):
    """Remove this tenant's previously stored chunks (t:{org}:doc:* keys)
    so re-ingestion leaves no stale data."""
    cursor = 0
    removed = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=f"{t.prefix}doc:*", count=500)
        if keys:
            r.delete(*keys)
            removed += len(keys)
        if cursor == 0:
            break
    return removed


def create_redis_index(r: Redis, t: TenantContext, dim: int) -> bool:
    """Create the tenant's FT index (`t.doc_index`) if missing or outdated;
    returns True when (re)created.

    Schema:
      text            TEXT     - full-text searchable chunk content
      embedding       VECTOR   - FLOAT32 / FLAT / COSINE, `dim` dimensions
      source_file     TAG      - exact-match filterable metadata
      section_heading TEXT     - filterable metadata
      chunk_pos       NUMERIC  - filterable metadata

    The index covers keys under the tenant's `t:{org}:doc:` prefix. Hash data
    is left in place on recreation - RediSearch rebuilds the index from
    existing keys automatically.
    """
    def find_value(obj, key):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kk = k.decode() if isinstance(k, bytes) else k
                if kk.lower() == str(key).lower():
                    return v
                found = find_value(v, key)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = find_value(item, key)
                if found is not None:
                    return found
        return None

    def needs_recreate() -> str | None:
        try:
            info = r.execute_command("FT.INFO", t.doc_index)
        except Exception:
            return "missing"
        if find_value(info, "dim") != dim:
            return "dim"
        # Step 6 migration: source_file must be TAG (was TEXT pre-hybrid-search)
        attrs = None
        if isinstance(info, dict):
            for k, v in info.items():
                kk = k.decode() if isinstance(k, bytes) else k
                if kk == "attributes":
                    attrs = v
                    break
        if isinstance(attrs, list):
            for attr in attrs:
                ident = find_value(attr, "identifier")
                if ident == "source_file":
                    ftype = find_value(attr, "type")
                    if ftype != "TAG":
                        return f"source_file type={ftype}"
        return None

    reason = needs_recreate()
    if reason is None:
        return False
    print(f"Recreating index '{t.doc_index}': {reason}")

    try:
        r.execute_command("FT.DROPINDEX", t.doc_index)
    except Exception:
        pass

    # FT.CREATE syntax: after FLAT comes the number of argument tokens that
    # follow (6 = TYPE + DIM + DISTANCE_METRIC pairs).
    r.execute_command(
        "FT.CREATE", t.doc_index,
        "ON", "HASH",
        "PREFIX", "1", f"{t.prefix}doc:",
        "SCHEMA",
        "text", "TEXT",
        "embedding", "VECTOR", "FLAT", "6",
            "TYPE", "FLOAT32",
            "DIM", str(dim),
            "DISTANCE_METRIC", "COSINE",
        "source_file", "TAG", "SEPARATOR", "|",
        "section_heading", "TEXT",
        "chunk_pos", "NUMERIC",
    )
    return True


def store_in_redis(r: Redis, t: TenantContext, chunks_with_emb) -> int:
    """Store each chunk as HSET {t.prefix}doc:{source_file}:{chunk_pos}.

    Keys live under the tenant's prefix, so tenants never share chunk data.
    The embedding is written as raw float32 bytes (required by RediSearch
    VECTOR fields); everything else is plain UTF-8 strings. Each chunk also
    carries `doc_version` (informational - not in the FT schema) recording
    which corpus version produced it.
    """
    stored = 0
    pipe = r.pipeline(transaction=False)
    for chunk in chunks_with_emb:
        emb_blob = np.asarray(chunk["embedding"], dtype=np.float32).tobytes()
        pipe.hset(
            f"{t.prefix}doc:{chunk['source_file']}:{chunk['chunk_pos']}",
            mapping={
                "text": chunk["chunk_text"],
                "embedding": emb_blob,
                "source_file": chunk["source_file"],
                "section_heading": chunk["heading"],
                "chunk_pos": str(chunk["chunk_pos"]),
                "doc_version": str(chunk.get("doc_version", 0)),
            },
        )
        stored += 1
        if stored % 200 == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)
    if stored % 200 != 0 or stored == 0:
        pipe.execute()
    return stored


def run_ingestion(r: Redis, t: TenantContext, docs_dir: str | Path = None,
                  verbose: bool = False) -> dict:
    """Full pipeline for one tenant; returns a summary dict for API/CLI use.

    Chunks are keyed under `t.prefix` and indexed in the tenant's own
    FT index (`t.doc_index`).
    """
    docs_dir = str(docs_dir or DOCS_DIR)
    t0 = time.time()
    events_mod.publish(r, t, events_mod.INGEST_STARTED,
                       {"tenant": t.id, "docs_dir": str(docs_dir)})

    def log(msg):
        if verbose:
            print(msg, flush=True)

    log("[1/4] Loading documentation files...")
    docs = collect_docs(docs_dir, t)
    log(f"      {len(docs)} files")

    log("[2/4] Chunking documents...")
    chunks = load_and_chunk_docs(docs)
    log(f"      {len(chunks)} chunks")

    log("[3/4] Embedding chunks...")
    model = FastInt8Embedder()
    texts = [c["chunk_text"] for c in chunks]
    embeddings = model.embed(texts, batch_size=64)
    log(f"      {len(embeddings)} vectors ({time.time() - t0:.1f}s elapsed)")

    # Version the corpus before storage so every chunk is stamped with the
    # version it belongs to (no-op when ENABLE_DOC_VERSIONING=0).
    prep = versions_mod.prepare(r, t, docs)

    log("[4/4] Storing in Redis...")
    removed = delete_source_keys(r, t)
    created = create_redis_index(r, t, EMBEDDING_DIM)
    chunks_with_emb = [
        {**c, "embedding": e, "doc_version": prep["version"]}
        for c, e in zip(chunks, embeddings)
    ]
    stored = store_in_redis(r, t, chunks_with_emb)
    version_summary = versions_mod.finalize(r, t, prep, chunks)

    duration = round(time.time() - t0, 2)
    summary = {
        "tenant": t.id,
        "documents": len(docs),
        "chunks_generated": len(chunks),
        "chunks_stored": stored,
        "stale_keys_removed": removed,
        "index_created": created,
        "index": t.doc_index,
        "embedding_dim": EMBEDDING_DIM,
        **version_summary,
        "duration_s": duration,
    }
    log(f"done in {duration}s")
    events_mod.publish(r, t, events_mod.INGEST_COMPLETED, summary)
    return summary
