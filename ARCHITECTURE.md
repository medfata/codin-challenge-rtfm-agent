# RTFM For Me Agent - Architecture

## Stack decisions (Step 0)

- **Language:** Python 3.13
- **Redis:** Single Redis 8 (`redis:8`) via Docker Compose (`docker-compose.yml`, port 6379, volume-backed). Search/JSON modules are built into Redis 8 natively. RedisInsight UI on port 8001. (Originally Redis Stack 7.x; consolidated in Aug 2026 because AMS requires the Redis 8.0 `HSETEX` command and no Stack image ships Redis 8.)
- **LLM:** tri-lane OpenAI-compatible strategy (Step 12) - generation **Groq** (`openai/gpt-oss-120b`) for doc answers, fast lane `openai/gpt-oss-20b` for routing/rewrite/summary, economy lane `gemini-3.5-flash-lite` -> Groq `qwen/qwen3.6-27b` failover for background/stored content (chitchat, memory synthesis, cache warming); automatic **Gemini** fallback on HTTP 429/5xx/connection errors (`ENABLE_LLM_FALLBACK`)
- **Embeddings:** local fastembed `BAAI/bge-small-en-v1.5` (384 dims - matches `FT.CREATE ... DIM`)
- **Redis client:** `redis-py` (async, env-driven `REDIS_URL`)
- **Sample docs:** Pro Git book, 91 AsciiDoc sections in `docs/progit2/`
- **Config:** `.env` (gitignored) + `.env.example`; keys: `GOOGLE_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `REDIS_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `TENANTS`
- **Verification:** `scripts/step0_check.py` - 5/5 checks (PING, FT._LIST, vector KNN, LLM chat, embeddings); `scripts/step7_check_routing.py` - routing, actions, fail-open drills

## What was done in Step 0

- Installed Docker Desktop (4.87.0, engine 29.7.2, Compose v5.4.0); fixed missing VC++ redistributable
- Scoped disk space first: C: 19.1 GB free, D: 8.2 GB free; install cost ~5.9 GB on C:
- `git init` + `.gitignore` + `pyproject.toml` + venv
- `docker-compose.yml` running `redis/redis-stack:latest` with persistent volume
- Verified Redis vector search end-to-end (FT.CREATE FLAT/COSINE index + KNN query)
- Switched LLM Grok -> Gemini after xAI 403 (no credits); verified Gemini chat works
- Verified fastembed 384-dim embeddings (model cached locally, first run downloads ~100 MB)
- Cloned Pro Git book sample docs (91 `.asc` files)

## Step 7 - Semantic routing (post-challenge feature)

Every message first passes through ONE fused LLM intent call (`gemini-3.5-flash`, temperature 0, JSON mode) that classifies the route and - for doc questions - produces the standalone rewritten search query plus an optional source-file hint (this retires the old `SOURCE:`-line text convention).

| Route | Behavior |
|---|---|
| `doc` | Full existing pipeline, unchanged (also the fail-open default) |
| `chitchat` | Persona micro-reply; skips cache/RAG/memory-search; turn kept in session history + long-term working memory |
| `memory` | Recalls long-term memories and synthesizes a personal answer ("what do you know about me?") |
| `action` | Deterministic handlers, no LLM: `metrics`, `list_docs`, `clear_session`, `flush_cache`, `reingest` |

Safety model:
- Malformed JSON, LLM errors, or unknown route/action values degrade to `doc` with the original question.
- Destructive actions (`flush_cache`, `reingest`) execute only when the raw question ALSO matches a keyword regex - the LLM verdict alone is never sufficient - and respect `ENABLE_DESTRUCTIVE_ACTIONS`.
- `reingest` runs as a background thread and acknowledges immediately.

Config: `ENABLE_ROUTING`, `ENABLE_ACTIONS`, `ENABLE_DESTRUCTIVE_ACTIONS`.
Verification: `scripts/step7_check_routing.py`.

Known tradeoffs:
- Routing happens before the semantic-cache decision, so even cache hits pay one small LLM call (~300-800 ms).
- Non-doc routes stay out of the cache hit/miss stats on purpose (hit_rate remains a pure doc-pipeline KPI); they still count toward `requests_total` and the `route_*_total` counters.
- SSE contract change for clients: chitchat/memory/action streams emit `route` then `token`(s) with no preceding `citations` event; every stream still ends with `done`.

## Step 8 - Multi-tenancy

Every request except `/health` must carry an `X-Tenant-Id` slug (`^[a-z0-9][a-z0-9-_]{0,62}$`, lowercased): missing/malformed -> `422`, valid slug outside the allowlist -> `403`. The validated id scopes all storage - Redis keys and FT indexes are prefixed `t:{org}:` (docs, cache, sessions, metrics) and Agent Memory Server records are isolated via `namespace={org}` (a rejected namespace filter yields empty results, never an unfiltered search). Ingestion prefers `docs/<tenant>/` when present; existing single-tenant data can be moved into a tenant with `scripts/migrate_to_tenant.py`.

Config: `TENANTS` (`acme,globex`, or `*` for open mode - any valid slug).
Verification: `scripts/step8_check_multitenancy.py`.

Safety/tradeoff: the header is trusted identity, not authentication - it carries the same trust level as the rest of the deployment. Strict mode breaks clients that omit the header (existing ones were updated), and the 2 extra FT indexes per tenant are fine for tens of teams but worth revisiting at hundreds.

## Step 9 - Document versioning

Each ingestion sha256-hashes every `.asc` file and folds the sorted `path:hash` pairs into a per-tenant corpus digest at `t:{org}:corpus`; the record's monotonic `version` increments only when the digest changes, so identical re-ingests invalidate nothing. Per-file metadata lives at `t:{org}:docmeta:{file}` and every chunk hash carries an informational `doc_version`. Staleness then has two detection points: semantic-cache entries are stamped with the corpus version that produced them, so hits generated under an older version are served **with** a warning (`stale: true` + text in `/ask`, a `warning` SSE event before tokens, counted in `stale_answers_served`) - never silently; and a TTL-cached on-disk hash scan compares live files against the ingest snapshot to power inline drift warnings, the `list_docs` action notice, and `GET /docs/status` (changed/added/removed detail). A missing corpus record means unversioned legacy data - behaviour identical to pre-Step 9, no warnings.

Config: `ENABLE_DOC_VERSIONING`, `ENABLE_DRIFT_WARNING`, `DRIFT_SCAN_TTL_S`.
Verification: `scripts/step9_check_versioning.py`.

Out of scope / next step: real-time Pub/Sub or Streams notifications when ingestion completes - built in Step 11 on top of these records.

Tradeoffs: warn-don't-block keeps stale cached answers one visible warning away instead of forcing regeneration; drift scanning re-hashes the docs dir only once per `DRIFT_SCAN_TTL_S` per tenant; auto content-hash versioning has no explicit labels or git SHAs; session-history answers are not retroactively flagged.

## Step 10 - MCP exposure

The assistant is also an MCP server: a streamable-HTTP `MCPServer` mounted at `/mcp` inside the same FastAPI app (stateless + JSON mode), so external agents (Claude Code/Desktop remote connectors, Cursor, any MCP HTTP client) call it as a tool set. Five tools, all resolving their tenant from the request's `X-Tenant-Id` header with `MCP_DEFAULT_TENANT` as fallback and reusing the pipeline internals in-process (no self-HTTP): `ask_question` (full RAG incl. routing + staleness, session-carrying), `search_documents` (raw KNN chunks), `list_documents`, `documentation_status`, `service_metrics` - plus a `docs://status` resource. Destructive actions stay REST-only. Optional shared-secret gate via `MCP_BEARER_TOKEN`; `MCP_ALLOWED_HOSTS` feeds the SDK's DNS-rebinding allowlist when deployed behind a real hostname. The mounted sub-app's lifespan never runs under `Mount`, so the host lifespan enters the MCP session manager explicitly.

Config: `ENABLE_MCP`, `MCP_DEFAULT_TENANT`, `MCP_BEARER_TOKEN`, `MCP_ALLOWED_HOSTS`.
Verification: `scripts/step10_check_mcp.py`.

Tradeoffs: HTTP-only v1 (no stdio wrapper yet); bearer token is a shared secret rather than per-agent identity; stateless mode drops cross-reconnect resumability; long-term memory is intentionally not proxied - agents talk to the Agent Memory Server's own native MCP endpoint side-by-side.

## Step 11 - Real-time event notifications

Every tenant owns one Redis Stream, `t:{org}:events`, acting as a MAXLEN-trimmed (exact trim, 1000 entries) ring buffer. Producers XADD small JSON envelopes (`type`, `ts`, `data`) from inside the existing pipelines: `ingest.started`/`ingest.completed` around `run_ingestion` (CLI and background reingest emit too), `ingest.failed` from the `/ingest` except-path and the reingest action thread, and `memory.turn_stored` after each successful working-memory save. Publishing is fail-open - event failures never break ingest/ask/memory paths.

Browsers subscribe via `GET /events/stream` (SSE): stream entry ids double as SSE ids, so a reconnecting client resumes exactly where it left off from its `Last-Event-ID` header (at-least-once; clients dedupe by id; malformed ids are ignored and fall back to the live tail instead of poisoning the stream), and `?backlog=N` replays the N most recent entries on a fresh connect. Because native `EventSource` cannot set headers, `?tenant=` is accepted as an equivalent trusted identity (same validation as `X-Tenant-Id`; sending both with different values is a 422). The subscriber loop is fully async - one shared `redis.asyncio` pool, blocking XREAD awaited on the event loop, so each stream costs zero worker threads and no dedicated socket; the pool's `socket_timeout` must exceed the block window (redis-py 8.x async otherwise falls back to the 5s connect timeout for reads, causing disconnect/reconnect churn). Heartbeat comments every ~15 s keep proxies alive and bound disconnect detection latency. Cross-origin browser clients need `EVENTS_CORS_ORIGINS` (GET/OPTIONS + `Last-Event-ID`/`X-Tenant-Id` headers only); the demo is also served same-origin at `/demo/events`.

Streams were chosen over Pub/Sub for durability + replay in one mechanism; Pub/Sub's fire-and-forget would lose events during disconnects/restarts. Publishing pipelines XADD with its metrics counter (one roundtrip) and is fail-open - event failures never break ingest/ask/memory paths. Known limits: `memory.turn_stored` fires only after AMS accepts the write (<400), and signals a saved turn, not long-term memories (AMS extracts those asynchronously in its own worker, no hooks).

Config: `ENABLE_EVENTS`, `EVENTS_STREAM_MAXLEN`, `EVENTS_HEARTBEAT_S`, `EVENTS_CORS_ORIGINS`.
Verification: `scripts/step11_check_events.py` (incl. a 40-subscriber concurrency drill); live feed demo at `scripts/events_demo.html` or `/demo/events`.

## Step 12 - Multi-model strategy

Redis doesn't care which model generated the content it stores, so LLM spend is split by task visibility: background/stored work rides a cheap **economy lane** (`gemini-3.5-flash-lite`, failing over to Groq `qwen/qwen3.6-27b` - a separate free pool, with `reasoning_effort: none`, so warm bursts never starve the gpt-oss-20b routing lane), while only the user-facing doc answer uses `gpt-oss-120b`. Chitchat replies and memory-route synthesis moved to economy; routing/rewrite/summary stay fast. The semantic cache gains **warming**: a per-tenant question-popularity ZSET (`{prefix}qfreq`, top-200, 30-day TTL) feeds a locked background job (`POST /cache/warm` or the conversational `warm_cache` action) that pre-answers popular questions with the economy lane straight into the cache, stamped with the current corpus version; live misses still generate with the capable model. AMS memory extraction switched to flash-lite (embeddings untouched). MCP inherits everything via `_pipeline`.

Config: `LLM_ECONOMY_MODEL`, `LLM_ECONOMY_FALLBACK_MODEL`, `ENABLE_CACHE_WARM`, `CACHE_WARM_TOP_N`, `QFREQ_TTL_S`, `QFREQ_MAX_ENTRIES`.
Verification: `scripts/step12_check_multi_model.py`.

Tradeoffs: warm answers are standalone (no session/persona context); chitchat/synthesis ride a small model in exchange for provider-outage resilience; AMS has no native fallback so Google-quota exhaustion pauses memory promotion until reset (escape hatch documented in docker-compose).

## Step 13 - Web crawl with staged review

`POST /crawl` discovers documentation on a website: sitemap.xml fast-path (one level of sitemap-index expansion) first, then a same-host breadth-first link frontier with optional `path_prefix` constraint, bounded by `max_pages`/`max_depth` (hard-capped by `CRAWL_HARD_PAGE_CAP`). Every request passes the SSRF guard (DNS-resolve the host; private/loopback/link-local/reserved targets are rejected unless `CRAWL_ALLOW_PRIVATE_HOSTS=1`, which exists for tests only), robots.txt compliance, politeness delay, response-size and content-type checks; trafilatura extracts readable markdown plus the page title.

Pages are **staged** under `docs/web/_staging/<org>/<job>/` - nothing touches the corpus until a human verifies them. The review UI at `/crawl/review` is a single self-contained static page (vanilla JS, served same-origin by FastAPI): it starts crawls, lists staged jobs, previews each page's extracted text, and approves/discards; live updates ride the Step 11 event stream (`crawl.staged` / `crawl.failed` alongside reused `ingest.*`). Approval (`POST /crawl/jobs/{id}/approve`, optional `exclude` list) copies kept pages to `docs/web/<org>/<host>/<slug>-<pid>.md` (leading `== Title` line so the standard loader picks up headings), deletes previously-approved files for vanished/excluded pages on covered hosts, then triggers a normal `run_ingestion()` - one merged corpus, so versioning, stale-cache warnings, drift scanning, hybrid search and citations all apply unchanged (web chunks are namespaced `web/<host>/<page>.md`). `auto_ingest=true` skips the review gate for trusted sources. Job lifecycle lives in `t:{org}:crawl:{job_id}` + a ZSET index; unreviewed staging sweeps after `CRAWL_STAGE_TTL_H`.

Config: `ENABLE_WEB_CRAWL`, `WEB_DOCS_DIR`, `CRAWL_MAX_PAGES`, `CRAWL_MAX_DEPTH`, `CRAWL_HARD_PAGE_CAP`, `CRAWL_DELAY_MS`, `CRAWL_TIMEOUT_S`, `CRAWL_MAX_BYTES`, `CRAWL_MIN_TEXT_CHARS`, `CRAWL_ALLOW_PRIVATE_HOSTS`, `CRAWL_STAGE_TTL_H`.
Verification: `scripts/step13_check_crawl.py`.

Tradeoffs: in-process background thread (no durable queue), one crawl per tenant at a time; approval runs a full-corpus re-ingest synchronously (same semantics as `POST /ingest`); extracted text approximates the source formatting; cross-host documentation sites need a future allowlist parameter.

## Step 13 - Web crawl with staged review

`POST /crawl` discovers documentation on a website: sitemap.xml fast-path (one level of sitemap-index expansion) first, then a same-host breadth-first link frontier with optional `path_prefix` constraint, bounded by `max_pages`/`max_depth` (hard-capped by `CRAWL_HARD_PAGE_CAP`). Every request passes the SSRF guard (DNS-resolve the host; private/loopback/link-local/reserved targets are rejected unless `CRAWL_ALLOW_PRIVATE_HOSTS=1`, which exists for tests only), robots.txt compliance, politeness delay, response-size and content-type checks; trafilatura extracts readable markdown plus the page title.

Pages are **staged** under `docs/web/_staging/<org>/<job>/` - nothing touches the corpus until a human verifies them. The review UI at `/crawl/review` is a single self-contained static page (vanilla JS, served same-origin by FastAPI): it starts crawls, lists staged jobs, previews each page's extracted text, and approves/discards; live updates ride the Step 11 event stream (`crawl.staged` / `crawl.failed` alongside reused `ingest.*`). Approval (`POST /crawl/jobs/{id}/approve`, optional `exclude` list) copies kept pages to `docs/web/<org>/<host>/<slug>-<pid>.md` (leading `== Title` line so the standard loader picks up headings), deletes previously-approved files for vanished/excluded pages on covered hosts, then triggers a normal `run_ingestion()` - one merged corpus, so versioning, stale-cache warnings, drift scanning, hybrid search and citations all apply unchanged (web chunks are namespaced `web/<host>/<page>.md`). `auto_ingest=true` skips the review gate for trusted sources. Job lifecycle lives in `t:{org}:crawl:{job_id}` + a ZSET index; unreviewed staging sweeps after `CRAWL_STAGE_TTL_H`.

Config: `ENABLE_WEB_CRAWL`, `WEB_DOCS_DIR`, `CRAWL_MAX_PAGES`, `CRAWL_MAX_DEPTH`, `CRAWL_HARD_PAGE_CAP`, `CRAWL_DELAY_MS`, `CRAWL_TIMEOUT_S`, `CRAWL_MAX_BYTES`, `CRAWL_MIN_TEXT_CHARS`, `CRAWL_ALLOW_PRIVATE_HOSTS`, `CRAWL_STAGE_TTL_H`.
Verification: `scripts/step13_check_crawl.py`.

Tradeoffs: in-process background thread (no durable queue), one crawl per tenant at a time; approval runs a full-corpus re-ingest synchronously (same semantics as `POST /ingest`); extracted text approximates the source formatting; cross-host documentation sites need a future allowlist parameter.

## LLM lanes & quotas (verified Aug 2026)

| Lane | Model | Free-tier ceiling |
|---|---|---|
| Fast (routing/rewrite/summary) | `openai/gpt-oss-20b` | 30 RPM / 1,000 req/day / 200K tok/day |
| Generation (final answers) | `openai/gpt-oss-120b` | 30 RPM / 1,000 req/day / 200K tok/day |
| Economy (stored/background) | `gemini-3.5-flash-lite` | ~15-30 RPM / ~1,000 req/day (flash family shares the daily pool) |
| Economy fallback | `qwen/qwen3.6-27b` | ~30 RPM / 1,000 req/day (separate Groq pool, `reasoning_effort: none`) |
| Fallback (generation/fast terminal) | `gemini-2.5-flash` | Google AI Studio free tier (limits visible per-project in AI Studio) |

Quotas are enforced per lane, so the fast lane never competes with generation traffic. Failover triggers on 429, 5xx, and connection errors; hard 4xx responses are surfaced immediately instead of burning the fallback lane.

## Diagram

```mermaid
flowchart LR
    subgraph INGEST["INGESTION (Steps 1+13)"]
        DOCS[Documentation<br/>.asc files] --> CHUNK[Chunker<br/>~500 tokens + overlap]
        WEB[Crawled pages .md<br/>docs/web/org - approved<br/>via /crawl/review] --> CHUNK
        CHUNK --> FE1[fastembed<br/>bge-small-en<br/>384 dims]
        FE1 --> VIDX[(REDIS - Docs index<br/>chunks + embedding +<br/>metadata: file, heading, pos)]
    end

    subgraph QUERY["QUESTION FLOW (Steps 2-7)"]
        USER[User] -->|question| API[REST API<br/>FastAPI]
        SESS[(REDIS - Sessions<br/>messages + TTL)] --> CTX[Context assembly]
        API --> INTENT{Intent router<br/>one fused LLM call:<br/>route + query + source}
        INTENT -->|chitchat| CHAT[Persona reply<br/>no retrieval]
        INTENT -->|memory| MEMR[Recall memories<br/>synthesize reply]
        INTENT -->|action| ACT[Action handlers<br/>guarded destructive ops]
        INTENT -->|doc| CACHE[(REDIS - Semantic cache<br/>question embedding + answer)]
        CACHE -->|hit - skip generation| OUT[Answer]
        CACHE -->|miss| RET[KNN top-k<br/>hybrid scoped]
        VIDX --> RET
        RET --> CTX
        MEM[Agent Memory Server<br/>long-term memories] --> CTX
        MEM -.->|recall| MEMR
        CTX --> LLM[LLM lanes<br/>gen: gpt-oss-120b<br/>economy: flash-lite -> llama-8b<br/>fallback: Gemini flash]
        LLM --> OUT
        OUT --> CACHE
        OUT --> SESS
        OUT -->|answer + citations| USER
    end

    MET[(REDIS - Metrics<br/>INCR counters)] -.-> API
    EVT[(REDIS - Events stream<br/>t:{org}:events)] -.->|SSE /events/stream| FE[Frontend<br/>EventSource]
    INGEST -.->|started/completed/failed| EVT
    SESS -.->|memory.turn_stored| EVT
    CFG[.env<br/>GOOGLE_API_KEY, REDIS_URL...] -.-> API
```

## Component map

| Component | Role | Status |
|---|---|---|
| Redis 8 (Docker) | vector search, cache, sessions, metrics, AMS memory | Step 0 done |
| fastembed | embeddings for chunks + queries | Step 0 done |
| Groq gpt-oss-120b (+ Gemini fallback) | answer generation + intent router | Step 7.1 |
| docs/progit2 | sample documentation | Step 0 done |
| Chunker + indexer | ingestion pipeline | Step 1 |
| REST API | /ingest, /ask, /metrics | Step 2-3 |
| Semantic cache | skip LLM on similar questions | Step 3 |
| Session memory | follow-up context, TTL | Step 4 |
| Agent Memory Server | long-term user memory | Step 5 |
| Hybrid search + hardening | metadata filters, summarisation | Step 6 |
| Semantic router | fused intent call: route + rewrite + source hint (JSON mode) | Step 7 |
| Action handlers | conversational ops: metrics, docs listing, session/cache ops | Step 7 |
| Multi-tenant scoping | X-Tenant-Id -> prefixed keys/indexes + AMS namespaces | Step 8 |
| Document versioning | content-hash corpus versions, stale-cache + drift warnings | Step 9 |
| MCP exposure | assistant as MCP tool server at /mcp (5 tools + resource) | Step 10 |
| Multi-model strategy | tri-lane LLM routing (generation/fast/economy) + economy-lane cache warming | Step 12 |
| Real-time events | per-tenant Redis Streams -> SSE feed with Last-Event-ID replay | Step 11 |
| Web crawl + review UI | sitemap/BFS discovery -> staged review at /crawl/review -> approved merge into tenant corpus | Step 13 |
| Web crawl + review UI | sitemap/BFS discovery -> staged review at /crawl/review -> approved merge into tenant corpus | Step 13 |