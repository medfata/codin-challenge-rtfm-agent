# Document Versioning Plan (Step 9)

Track per-tenant document versions via automatic content hashing and warn users when an answer reflects outdated documentation. No manual version labels, no incremental re-ingest.

**Out of scope (next step):** Real-time update notifications via Redis Pub/Sub/Streams on ingestion completion - the versioning records built here are its foundation.

## Version model

| Record | Key | Fields |
|---|---|---|
| Corpus record | `t:{org}:corpus` (HASH) | `version` (int, bumped **only when digest changes**), `digest` (sha256 over sorted `path:contenthash` lines), `files`, `chunks_total`, `ingested_at`, `file_hashes_json` |
| Per-doc metadata | `t:{org}:docmeta:{source_file}` (HASH) | `sha256`, `chunks`, `ingested_at` |
| Chunk stamps | existing chunk hashes gain `doc_version` field | informational only - no FT index/schema change |

Backward compatible: absent corpus record -> behavior identical to today (no warnings). Unversioned cache entries count as stale once corpus version >= 1. Identical-content re-ingests do not bump the version or invalidate anything.

## Detection & surfacing

| Staleness source | Mechanism | Surface |
|---|---|---|
| Cache hit from older corpus | cache entries stamped with `corpus_version`; compared at `lookup()` | Policy **serve-with-warning**: `stale: true` + `version_warning` in `/ask` JSON; new `warning` SSE event before tokens; `stale_answers_served` metric |
| Disk drifted since last ingest | hashed dir scan vs ingest-time hashes, TTL-cached per tenant+dir (mirrors `scope._inventory_cache`) | Inline answer-time warning + `GET /docs/status` endpoint + drift notice in `list_docs` action |

Warning copy composed once in `versions.py`, shared by REST + SSE:
- "This answer was cached from documentation v3; the corpus is now at v5 (re-ingested <ts>). Ask me to flush the cache or re-ingest for a fresh answer."
- "N indexed documents have changed on disk since the last ingestion (v4, <ts>); answers reflect the stored copies until re-ingest."

## Touchpoints

| File | Change |
|---|---|
| `rtfm_agent/versions.py` (new) | file hashing, digest, `prepare()`/`finalize()` around chunk storage (conditional bump + docmeta upserts), `get_corpus()`, TTL-cached `scan_drift()`, warning composers |
| `rtfm_agent/config.py` | `ENABLE_DOC_VERSIONING=1`, `ENABLE_DRIFT_WARNING=1`, `DRIFT_SCAN_TTL_S=30` |
| `rtfm_agent/ingest.py` | stamp chunks with `doc_version`; prepare/finalize calls; summary gains `{corpus_version, digest, added, updated, removed, unchanged}` |
| `rtfm_agent/cache.py` | `store()` persists `corpus_version`; `lookup()` returns it (fail-open) |
| `rtfm_agent/api.py` | compose warnings in both pipelines; `AskResponse.stale/warning`; SSE `warning` event + fields in `done`; `/docs/status` route |
| `rtfm_agent/actions.py` | `list_docs` shows corpus version + drift; `metrics` report gains stale-answer line |
| `rtfm_agent/metrics.py` | `stale_answers_served` counter |
| `ARCHITECTURE.md` | Step 9 section, diagram note, component-map row |
| `scripts/step9_check_versioning.py` (new) | verification drill, zero LLM calls |

## Test plan

1. Hash determinism: identical content -> identical digest; any edit -> different digest
2. First ingest -> corpus v1 + one `docmeta` per file; identical re-ingest keeps v1, reports `unchanged`
3. Edit one file -> re-ingest -> v2, `updated: 1`, docmeta refreshed, chunk stamps carry `doc_version=2`
4. Cache drill: entry stored at v2 served at v3 -> stale detected with warning; unversioned entry stale once v >= 1; post-flush clean
5. Drift drill: edit file without re-ingesting -> drift report lists it; re-ingest clears
6. Legacy-data drill: wipe corpus/docmeta keys -> no crashes, no warnings
7. Tenant isolation: A's corpus/docmeta invisible to B; drift scan scoped per tenant dir
8. Key hygiene: all new keys under `t:<slug>:`

## Tradeoffs

- Warn-don't-block: stale cached answers still served (mitigated by visible warning + existing `flush_cache` action)
- Drift scan cost bounded by TTL cache; very large repos may want git-based detection later
- Auto-hash only: no git SHA / explicit labels (zero maintenance chosen over richer provenance)
- Session-history answers not retroactively flagged - only fresh responses and cache hits carry staleness info
- Incremental ingestion deferred; hashing infrastructure makes it a small follow-up
- Real-time Pub/Sub notifications deferred to the next step
