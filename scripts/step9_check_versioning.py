"""Step 9 verification: document versioning + stale-answer warnings.

Part A: pure unit checks (hash/digest determinism, warning composers), no services.
Part B: Redis drills on synthetic tenants - version lifecycle across repeated
        ingests, cache stamping/staleness, disk-drift detection, legacy-data
        tolerance, tenant isolation, key hygiene. No LLM, no embedder -
        chunk embeddings are fabricated zeros because nothing here runs
        vector searches.
Part C: live HTTP probe of /docs/status against RTFM_API_URL (skipped when
        the API is unreachable).

Redis drills and API probes degrade to [SKIP]; exit code 1 iff any FAIL.
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from redis import Redis

from rtfm_agent import cache as cache_mod
from rtfm_agent import ingest as ingest_mod
from rtfm_agent import metrics as metrics_mod
from rtfm_agent import versions as versions_mod
from rtfm_agent.config import EMBEDDING_DIM, REDIS_URL
from rtfm_agent.documents import load_asc_files
from rtfm_agent.tenancy import TenantContext

PASS = 0
FAIL = 0

TENANT = "step9"
TENANT_B = "step9b"
API_BASE = os.getenv("RTFM_API_URL", "http://localhost:8000").rstrip("/")

FILES_V1 = {
    "alpha.asc": "# Alpha\n\nAlpha covers vectors.\n\n== Vectors\n\nVectors are lists of numbers.",
    "beta.asc": "# Beta\n\nBeta covers caching.\n\n== Cache\n\nA cache avoids repeat work.",
    "gamma.asc": "# Gamma\n\nGamma covers memory.",
}
FILE_BETA_EDITED = (
    "# Beta\n\nBeta covers semantic caching.\n\n== Semantic cache\n"
    "\nEmbed questions and reuse near-identical answers."
)
FILE_DELTA_NEW = "# Delta\n\nDelta covers streaming."


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"[PASS] {name}" + (f" - {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"[FAIL] {name}" + (f" - {detail}" if detail else ""))


def skip(name: str, detail: str = "") -> None:
    print(f"[SKIP] {name}" + (f" - {detail}" if detail else ""))


def part_a_units() -> None:
    h1 = versions_mod.hash_content("hello world")
    h2 = versions_mod.hash_content("hello world")
    h3 = versions_mod.hash_content("hello worlds")
    check("hash_content deterministic", h1 == h2)
    check("hash_content sensitive to edits", h1 != h3)

    d1 = versions_mod.compute_digest({"a.asc": h1, "b.asc": h3})
    d2 = versions_mod.compute_digest({"b.asc": h3, "a.asc": h1})
    d3 = versions_mod.compute_digest({"a.asc": h3, "b.asc": h1})
    check("compute_digest order-stable", d1 == d2)
    check("compute_digest changes when any file hash moves", d1 != d3)

    corpus = {"version": 3, "ingested_at": 0.0}
    check("cache_staleness_message None when current", 
          versions_mod.cache_staleness_message(3, corpus) is None)
    msg = versions_mod.cache_staleness_message(1, corpus) or ""
    check("cache_staleness_message names both versions",
          "v1" in msg and "v3" in msg, msg[:60])
    legacy = versions_mod.cache_staleness_message(0, corpus) or ""
    check("legacy unversioned entries get dedicated wording",
          "before document version tracking" in legacy)

    empty = {"changed": [], "added": [], "removed": []}
    drifted = {"changed": ["x"], "added": [], "removed": []}
    check("drift_message None when clean",
          versions_mod.drift_message(empty, corpus) is None)
    check("drift_message counts changes",
          "1 indexed documents" in (versions_mod.drift_message(drifted, corpus) or ""))


def _write_docs(docs_dir: Path, files: dict[str, str]) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (docs_dir / name).write_text(content, encoding="utf-8")


def _fake_chunks(docs, version: int) -> list[dict]:
    zeros = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return [
        {
            "source_file": d["source_file"],
            "heading": d["heading"],
            "chunk_pos": 1,
            "chunk_text": d["content"][:200],
            "embedding": zeros,
            "doc_version": version,
        }
        for d in docs
    ]


def _run_round(r: Redis, t: TenantContext, docs_dir: Path) -> tuple[dict, int]:
    docs = load_asc_files(str(docs_dir))
    prep = versions_mod.prepare(r, t, docs)
    ingest_mod.delete_source_keys(r, t)
    chunks = _fake_chunks(docs, prep["version"])
    ingest_mod.store_in_redis(r, t, chunks)
    summary = versions_mod.finalize(r, t, prep, chunks)
    return summary, prep["version"]


def _wipe_tenant(r: Redis, slug: str) -> None:
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=f"t:{slug}:*", count=500)
        if keys:
            r.delete(*keys)
        if cursor == 0:
            break


def part_b_redis() -> None:
    try:
        r = Redis.from_url(REDIS_URL, decode_responses=False, socket_connect_timeout=3)
        r.ping()
    except Exception as exc:
        skip("Redis drills", f"no Redis at {REDIS_URL}: {exc}")
        return

    t = TenantContext(TENANT)
    tb = TenantContext(TENANT_B)
    tmp = Path(tempfile.mkdtemp(prefix="step9_docs_", dir=str(project_root)))
    try:
        # --- Round 1: first ingestion -> corpus v1 -------------------------
        _write_docs(tmp, FILES_V1)
        summary, version = _run_round(r, t, tmp)
        check("first ingestion creates corpus v1", version == 1,
              f"v={version}")
        check("first ingestion reports all files as added",
              summary["added"] == 3 and summary["updated"] == 0
              and summary["unchanged"] == 0 and summary["removed"] == 0,
              json.dumps({k: summary[k] for k in
                          ("added", "updated", "unchanged", "removed")}))
        corpus = versions_mod.get_corpus(r, t)
        check("corpus record persisted with digest/files/chunks",
              corpus is not None and corpus["digest"]
              and corpus["files"] == 3 and corpus["chunks_total"] == 3,
              f"files={corpus['files'] if corpus else None}")
        meta_count = sum(1 for _ in r.scan_iter(match=f"{t.prefix}docmeta:*"))
        check("one docmeta hash per file", meta_count == 3, f"count={meta_count}")
        sample_meta = r.hgetall(versions_mod.docmeta_key(t, "alpha.asc"))
        check("docmeta carries sha256 + chunk count",
              bool(sample_meta.get(b"sha256")) and sample_meta.get(b"chunks") == b"1")
        chunk_fields = r.hgetall(f"{t.prefix}doc:alpha.asc:1")
        check("chunk hash stamped doc_version=1",
              chunk_fields.get(b"doc_version") == b"1")

        # --- Round 2: identical re-ingest keeps v1 --------------------------
        summary, version = _run_round(r, t, tmp)
        check("identical re-ingest does NOT bump version", version == 1,
              f"v={version}")
        check("identical re-ingest reports unchanged=3",
              summary["unchanged"] == 3 and summary["added"] == 0
              and summary["updated"] == 0)

        # --- Drift: edit beta.asc on disk WITHOUT re-ingesting --------------
        (tmp / "beta.asc").write_text(FILE_BETA_EDITED, encoding="utf-8")
        drift = versions_mod.scan_drift(r, t, tmp, force=True)
        check("drift detects edited file", drift["changed"] == ["beta.asc"],
              f"changed={drift['changed']}")
        check("drift report counts", versions_mod.drift_changed_count(drift) == 1)
        staleness = versions_mod.answer_staleness(
            r, t, str(tmp), cached_version=None)
        check("fresh answers warn while disk is ahead of the index",
              staleness["stale"] and "changed on disk" in (staleness["warning"] or ""),
              (staleness["warning"] or "")[:60])

        # --- Round 3: re-ingest the edit -> v2 ------------------------------
        summary, version = _run_round(r, t, tmp)
        check("edited re-ingest bumps to v2", version == 2, f"v={version}")
        check("edited re-ingest reports updated=1",
              summary["updated"] == 1 and summary["unchanged"] == 2)
        chunk_fields = r.hgetall(f"{t.prefix}doc:beta.asc:1")
        check("re-stored chunk carries new doc_version=2",
              chunk_fields.get(b"doc_version") == b"2")
        meta = r.hgetall(versions_mod.docmeta_key(t, "beta.asc"))
        fresh_sha = versions_mod.hash_content(FILE_BETA_EDITED).encode()
        check("docmeta sha256 refreshed after edit", meta.get(b"sha256") == fresh_sha)
        drift = versions_mod.scan_drift(r, t, tmp, force=True)
        check("drift clean after re-ingest", drift["changed"] == [])

        # --- Cache stamping + stale-hit detection ---------------------------
        cache_mod.ensure_cache_index(r, t)
        qvec = np.zeros(EMBEDDING_DIM, dtype=np.float32).tobytes()
        citations = [{"source_file": "beta.asc", "section_heading": "s",
                      "chunk_pos": 1, "score": 0.1}]
        cache_mod.store(r, t, "how does caching work?", qvec, "old answer",
                        citations, corpus_version=1)
        hit = cache_mod.lookup(r, t, qvec)
        check("cache hit carries its corpus_version",
              hit is not None and hit["corpus_version"] == 1,
              f"cv={hit.get('corpus_version') if hit else None}")
        staleness = versions_mod.answer_staleness(
            r, t, str(tmp), cached_version=hit["corpus_version"])
        check("cache hit from v1 is stale at v2 with warning",
              staleness["stale"]
              and "cached from documentation v1" in (staleness["warning"] or "")
              and "now at v2" in (staleness["warning"] or ""),
              (staleness["warning"] or "")[:80])

        # Legacy-style entry with no version field reads as 0 -> stale too.
        # Flush first: identical zero vectors tie in KNN, so a stale entry
        # from the previous sub-drill could win the lookup instead.
        cache_mod.flush(r, t)
        raw_key = cache_mod.store(r, t, "legacy entry", qvec, "older answer",
                                  citations)
        r.hdel(raw_key, "corpus_version")
        legacy_hit = cache_mod.lookup(r, t, qvec)
        check("unversioned cache entries read back as corpus_version 0",
              legacy_hit is not None and legacy_hit["corpus_version"] == 0)
        staleness = versions_mod.answer_staleness(
            r, t, str(tmp), cached_version=legacy_hit["corpus_version"])
        check("unversioned cache hits flagged stale once tracking exists",
              staleness["stale"]
              and "before document version tracking" in (staleness["warning"] or ""))

        # Current-version entries are never flagged.
        staleness = versions_mod.answer_staleness(
            r, t, str(tmp), cached_version=versions_mod.current_version(r, t))
        check("fresh cache hits at current version stay clean",
              not staleness["stale"] and staleness["warning"] is None)

        metrics_mod.record_stale_answer(r, t)
        snap = metrics_mod.snapshot(r, t)
        check("stale_answers_served counter wired into metrics snapshot",
              snap["stale_answers_served"] >= 1)

        # --- New file appears on disk without re-ingest ---------------------
        _write_docs(tmp, {"delta.asc": FILE_DELTA_NEW})
        drift = versions_mod.scan_drift(r, t, tmp, force=True)
        check("drift detects newly added disk file", drift["added"] == ["delta.asc"])

        # --- Status report (corpus v2, disk one file ahead) -----------------
        report = versions_mod.status_report(r, t, str(tmp))
        check("status_report exposes corpus + drift + up_to_date flag",
              report["corpus"]["version"] == 2
              and report["up_to_date"] is False
              and report["drift"]["added"] == ["delta.asc"],
              json.dumps(report["drift"].get("added")))

        # --- Round 4: file removed from corpus -> v3 + docmeta cleanup ------
        # (Round 4's re-ingest also absorbs delta.asc, clearing the drift.)
        os.remove(tmp / "gamma.asc")
        summary, version = _run_round(r, t, tmp)
        check("removal re-ingest bumps to v3", version == 3, f"v={version}")
        check("removal reported", summary["removed"] == 1)
        check("docmeta deleted for removed file",
              not r.exists(versions_mod.docmeta_key(t, "gamma.asc")))
        drift = versions_mod.scan_drift(r, t, tmp, force=True)
        check("re-ingest that absorbs new files clears the drift",
              versions_mod.drift_changed_count(drift) == 0)

        # --- Legacy-data tolerance -------------------------------------------
        r.delete(versions_mod.corpus_key(t))
        for key in r.scan_iter(match=f"{t.prefix}docmeta:*"):
            r.delete(key)
        check("missing corpus record reads as unversioned",
              versions_mod.get_corpus(r, t) is None
              and versions_mod.current_version(r, t) == 0)
        staleness = versions_mod.answer_staleness(
            r, t, str(tmp), cached_version=5)
        check("no stale warnings without a corpus record",
              not staleness["stale"] and staleness["warning"] is None)
        report = versions_mod.status_report(r, t, str(tmp))
        check("status_report tolerates unversioned state",
              report["corpus"] is None and report["up_to_date"] is True)

        # --- Tenant isolation + key hygiene ----------------------------------
        check("other tenant sees no corpus", versions_mod.get_corpus(r, tb) is None)
        allowed = (f"t:{TENANT}:".encode(), f"t:{TENANT_B}:".encode())
        stray = [k for k in r.scan_iter(match=b"*step9*")
                 if not k.startswith(allowed)]
        check("all step9 keys tenant-prefixed", not stray, f"stray={stray[:3]}")
    finally:
        _wipe_tenant(r, TENANT)
        _wipe_tenant(r, TENANT_B)
        shutil.rmtree(tmp, ignore_errors=True)


def part_c_http() -> None:
    import httpx

    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=3)
        alive = resp.status_code < 500
    except Exception:
        alive = False
    if not alive:
        skip("HTTP /docs/status probe", f"API not reachable at {API_BASE}")
        return

    headers = {"X-Tenant-Id": TENANT}
    try:
        resp = httpx.get(f"{API_BASE}/docs/status", headers=headers, timeout=10)
        body = resp.json()
        check("GET /docs/status returns corpus/drift envelope",
              resp.status_code == 200
              and {"tenant", "versioning_enabled", "corpus", "drift",
                   "up_to_date"} <= set(body),
              f"status={resp.status_code}")
    except Exception as exc:
        skip("HTTP /docs/status probe", f"request failed: {exc}")


if __name__ == "__main__":
    t0 = time.time()
    print(f"== Step 9 checks - {time.strftime('%Y-%m-%d %H:%M:%S')} ==")
    part_a_units()
    part_b_redis()
    part_c_http()
    print(f"\n{PASS} passed, {FAIL} failed in {time.time() - t0:.1f}s")
    sys.exit(1 if FAIL else 0)
