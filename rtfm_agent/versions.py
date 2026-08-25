"""Step 9: document versioning - content-hash corpus tracking + staleness warnings.

Each tenant's corpus carries an implicit version derived purely from file
content (no manual labels):

    t:{org}:corpus          HASH  version, digest, files, chunks_total,
                                  ingested_at, file_hashes_json
    t:{org}:docmeta:{file}  HASH  sha256, chunks, ingested_at

`digest` is a sha256 over the sorted "path:sha256" lines of every indexed
file, so any content change anywhere flips it. The numeric `version` only
increments when the digest changes - identical re-ingests are free.

Staleness surfaces:
  * cache hits stamped with an older corpus_version -> serve-with-warning
  * disk files edited after the last ingest -> drift warning (/docs/status,
    list_docs action, inline on /ask)

Everything here fails open: a missing/corrupt corpus record simply means
"unversioned", which produces no warnings and changes no behaviour.
"""

import hashlib
import json
import logging
import time
from pathlib import Path

from redis import Redis

from rtfm_agent.config import (
    DRIFT_SCAN_TTL_S,
    ENABLE_DOC_VERSIONING,
    ENABLE_DRIFT_WARNING,
)
from rtfm_agent.tenancy import TenantContext

logger = logging.getLogger(__name__)

# In-process drift-scan cache keyed by (tenant id, resolved docs dir):
# {(org, dir): (monotonic_ts, report)} - mirrors scope._inventory_cache.
_drift_cache: dict[tuple[str, str], tuple[float, dict]] = {}


def _field(data: dict, key: str):
    """Fetch a hash field regardless of decode_responses mode."""
    for k in (key, key.encode()):
        if k in data:
            v = data[k]
            return v.decode(errors="replace") if isinstance(v, bytes) else v
    return None


def hash_content(content: str) -> str:
    """sha256 hex digest of one file's full text."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_digest(file_hashes: dict[str, str]) -> str:
    """Corpus digest over sorted 'path:hash' pairs - order-stable."""
    payload = "\n".join(f"{p}:{h}" for p, h in sorted(file_hashes.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scan_disk(docs_dir: str | Path) -> dict[str, str]:
    """sha256 of every .asc file under docs_dir (recursive), posix relpaths."""
    base = Path(docs_dir)
    out: dict[str, str] = {}
    if not base.is_dir():
        return out
    for path in sorted(base.rglob("*.asc")):
        if ".git" in path.parts:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("drift scan could not read %s: %s", path, exc)
            continue
        out[path.relative_to(base).as_posix()] = hash_content(content)
    return out


def corpus_key(t: TenantContext) -> str:
    return f"{t.prefix}corpus"


def docmeta_key(t: TenantContext, source_file: str) -> str:
    return f"{t.prefix}docmeta:{source_file}"


def get_corpus(r: Redis, t: TenantContext) -> dict | None:
    """The tenant's corpus record, or None when never versioned / unreadable."""
    try:
        data = r.hgetall(corpus_key(t))
    except Exception as exc:
        logger.warning("corpus lookup failed (non-fatal): %s", exc)
        return None
    if not data:
        return None
    try:
        version = int(_field(data, "version") or 0)
    except ValueError:
        return None
    try:
        file_hashes = json.loads(_field(data, "file_hashes_json") or "{}")
    except json.JSONDecodeError:
        file_hashes = {}
    return {
        "version": version,
        "digest": _field(data, "digest") or "",
        "files": int(_field(data, "files") or 0),
        "chunks_total": int(_field(data, "chunks_total") or 0),
        "ingested_at": float(_field(data, "ingested_at") or 0.0),
        "file_hashes": file_hashes,
    }


def current_version(r: Redis, t: TenantContext) -> int:
    corpus = get_corpus(r, t)
    return corpus["version"] if corpus else 0


def prepare(r: Redis, t: TenantContext, docs: list[dict]) -> dict:
    """Pre-storage half of versioning: hash the incoming docs and decide the
    version this ingestion will carry.

    Returns {enabled, file_hashes, digest, version, diff, previous} where
    diff = {added, updated, unchanged} vs the previously recorded hashes.
    Writes nothing - safe to run before chunks are stored.
    """
    empty = {
        "enabled": False, "file_hashes": {}, "digest": "", "version": 0,
        "diff": {"added": 0, "updated": 0, "unchanged": 0}, "previous": None,
    }
    if not ENABLE_DOC_VERSIONING:
        return empty

    file_hashes = {d["source_file"]: hash_content(d["content"]) for d in docs}
    digest = compute_digest(file_hashes)
    previous = get_corpus(r, t)

    old_hashes: dict[str, str] = (previous or {}).get("file_hashes") or {}
    added = [p for p in file_hashes if p not in old_hashes]
    updated = [p for p in file_hashes if p in old_hashes and file_hashes[p] != old_hashes[p]]
    unchanged = [p for p in file_hashes if p in old_hashes and file_hashes[p] == old_hashes[p]]

    version = 1 if previous is None else previous["version"]
    if previous is not None and digest != previous["digest"]:
        version += 1

    return {
        "enabled": True,
        "file_hashes": file_hashes,
        "digest": digest,
        "version": version,
        "diff": {"added": len(added), "updated": len(updated), "unchanged": len(unchanged)},
        "previous": previous,
    }


def finalize(r: Redis, t: TenantContext, prep: dict, chunks: list[dict]) -> dict:
    """Post-storage half: persist the corpus record and per-doc metadata.

    Chunk counts come from the freshly stored chunk list. Docmeta keys for
    files that disappeared from the corpus are deleted. Returns the summary
    block merged into the ingestion response.
    """
    if not prep.get("enabled"):
        return {
            "corpus_version": 0, "digest": "",
            "added": 0, "updated": 0, "unchanged": 0, "removed": 0,
        }

    now = time.time()
    chunk_counts: dict[str, int] = {}
    for c in chunks:
        chunk_counts[c["source_file"]] = chunk_counts.get(c["source_file"], 0) + 1

    previous = prep.get("previous")
    old_paths = set((previous or {}).get("file_hashes") or {})
    removed_paths = sorted(old_paths - set(prep["file_hashes"]))

    try:
        pipe = r.pipeline(transaction=False)
        pipe.hset(corpus_key(t), mapping={
            "version": str(prep["version"]),
            "digest": prep["digest"],
            "files": str(len(prep["file_hashes"])),
            "chunks_total": str(len(chunks)),
            "ingested_at": str(now),
            "file_hashes_json": json.dumps(prep["file_hashes"]),
        })
        for path, sha in prep["file_hashes"].items():
            pipe.hset(docmeta_key(t, path), mapping={
                "sha256": sha,
                "chunks": str(chunk_counts.get(path, 0)),
                "ingested_at": str(now),
            })
        for path in removed_paths:
            pipe.delete(docmeta_key(t, path))
        pipe.execute()
    except Exception as exc:
        # Fail open: versioning metadata is advisory; chunks are already stored.
        logger.warning("version finalize failed (non-fatal): %s", exc)

    return {
        "corpus_version": prep["version"],
        "digest": prep["digest"],
        **prep["diff"],
        "removed": len(removed_paths),
    }


def scan_drift(r: Redis, t: TenantContext, docs_dir: str | Path,
               force: bool = False) -> dict:
    """Compare on-disk .asc hashes against the last ingestion's snapshot.

    TTL-cached in-process per (tenant, docs dir); `force` bypasses the cache
    (used by /docs/status and tests). With no corpus record there is nothing
    to compare against - the report comes back empty-but-valid.
    """
    try:
        resolved = str(Path(docs_dir).resolve())
    except OSError:
        resolved = str(docs_dir)
    cache_id = (t.id, resolved)
    now = time.monotonic()
    if not force:
        cached = _drift_cache.get(cache_id)
        if cached is not None and now - cached[0] < DRIFT_SCAN_TTL_S:
            return cached[1]

    empty = {
        "changed": [], "added": [], "removed": [],
        "disk_files": 0, "indexed_files": 0, "scanned_at": time.time(),
        "comparable": False,
    }
    corpus = get_corpus(r, t)
    indexed: dict[str, str] = (corpus or {}).get("file_hashes") or {}
    if corpus is None:
        _drift_cache[cache_id] = (now, empty)
        return empty

    disk = scan_disk(resolved)
    report = {
        "changed": sorted(p for p in disk if p in indexed and disk[p] != indexed[p]),
        "added": sorted(p for p in disk if p not in indexed),
        "removed": sorted(p for p in indexed if p not in disk),
        "disk_files": len(disk),
        "indexed_files": len(indexed),
        "scanned_at": time.time(),
        "comparable": True,
    }
    _drift_cache[cache_id] = (now, report)
    return report


def drift_changed_count(report: dict) -> int:
    return len(report.get("changed", [])) + len(report.get("added", [])) \
        + len(report.get("removed", []))


def cache_staleness_message(cached_version: int, corpus: dict | None) -> str | None:
    """Warning text for a cache hit generated under an older corpus.

    Version 0 means the entry predates version tracking entirely.
    """
    current = (corpus or {}).get("version") or 0
    if current < 1 or cached_version >= current:
        return None
    ingested = time.strftime(
        "%Y-%m-%d %H:%M UTC", time.gmtime((corpus or {}).get("ingested_at") or 0)
    )
    if cached_version < 1:
        return (
            f"This answer was generated before document version tracking was "
            f"enabled; the documentation corpus has since been updated to "
            f"v{current} ({ingested}). Re-ingesting or flushing the cache will "
            f"refresh it."
        )
    return (
        f"This answer was cached from documentation v{cached_version}; the "
        f"corpus is now at v{current} ({ingested}). Ask me to flush the cache "
        f"or re-ingest for a fresh answer."
    )


def drift_message(report: dict, corpus: dict | None) -> str | None:
    """Warning text for on-disk edits that postdate the last ingestion."""
    n = drift_changed_count(report)
    if n < 1:
        return None
    version = (corpus or {}).get("version") or 0
    ingested = time.strftime(
        "%Y-%m-%d %H:%M UTC", time.gmtime((corpus or {}).get("ingested_at") or 0)
    )
    return (
        f"{n} indexed documents have changed on disk since the last ingestion "
        f"(v{version}, {ingested}); answers reflect the stored copies until "
        f"re-ingest."
    )


def answer_staleness(r: Redis, t: TenantContext, docs_dir: str | None = None,
                     cached_version: int | None = None) -> dict:
    """Single composition point for /ask staleness info (REST + SSE).

    Pass `cached_version` for cache-served answers; omit it for fresh
    generation (only drift can apply). Returns {stale, warning}.
    """
    if not ENABLE_DOC_VERSIONING:
        return {"stale": False, "warning": None}
    try:
        corpus = get_corpus(r, t)
    except Exception as exc:
        logger.warning("staleness check failed (non-fatal): %s", exc)
        return {"stale": False, "warning": None}

    reasons: list[str] = []
    if cached_version is not None:
        msg = cache_staleness_message(cached_version, corpus)
        if msg:
            reasons.append(msg)
    if ENABLE_DRIFT_WARNING and docs_dir:
        try:
            msg = drift_message(scan_drift(r, t, docs_dir), corpus)
            if msg:
                reasons.append(msg)
        except Exception as exc:
            logger.warning("drift warning failed (non-fatal): %s", exc)

    return {"stale": bool(reasons), "warning": " ".join(reasons) or None}


def status_report(r: Redis, t: TenantContext, docs_dir: str | None = None) -> dict:
    """Payload for GET /docs/status: corpus state plus a forced drift scan."""
    corpus = get_corpus(r, t)
    report = {
        "tenant": t.id,
        "versioning_enabled": ENABLE_DOC_VERSIONING,
        "corpus": None,
        "drift": {},
        "up_to_date": True,
    }
    if corpus:
        report["corpus"] = {k: corpus[k] for k in
                            ("version", "digest", "files", "chunks_total", "ingested_at")}
    if ENABLE_DRIFT_WARNING and docs_dir:
        drift = scan_drift(r, t, docs_dir, force=True)
        report["drift"] = drift
        report["up_to_date"] = drift_changed_count(drift) == 0
    return report
