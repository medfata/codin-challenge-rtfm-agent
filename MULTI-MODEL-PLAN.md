# Multi-Model Strategy Plan (Step 12)

Use a cheaper model for memory extraction and caching, and a more capable
model for final answers. Redis doesn't care which model generates the content
it stores - so background/stored work rides a cheap economy lane, and only
the live user-facing answer burns the capable generation pool.

## Lane map

| Lane | Chain (auto-failover on 429/5xx/connection error) | Used for |
|---|---|---|
| `generation` | Groq `gpt-oss-120b` -> Google `gemini-2.5-flash` | doc-route final answers only |
| `fast` | Groq `gpt-oss-20b` -> Google `gemini-2.5-flash` | intent routing, query rewrite, session summary |
| `economy` (new) | Google `gemini-3.5-flash-lite` -> Groq `qwen/qwen3.6-27b` | chitchat replies, memory-route synthesis, cache warming |
| AMS extraction | `GENERATION_MODEL`/`FAST_MODEL` = `gemini/gemini-3.5-flash-lite` | memory extraction / working-memory ops |

Key quota insight: Gemini's free tier shares one daily pool across the
flash family, so falling back flash-lite -> flash is useless once exhausted -
the economy chain crosses providers instead. The Groq hop is
`qwen/qwen3.6-27b` (own pool, ~1k req/day; `llama-3.1-8b-instant` was
decommissioned on Groq in 2026) NOT `gpt-oss-20b`, so warm bursts can never
starve routing. Qwen3 emits `<think>` reasoning by default; the economy
lane sends `reasoning_effort: "none"` via a per-lane `payload_extra`. Note:
`gemini-2.5-flash-lite` was retired for new keys (Aug 2026) - the default is
`gemini-3.5-flash-lite`. When no Google key is configured the economy lane
silently starts on the Groq fallback.

## Cache warming

- Popularity: per-tenant ZSET `{prefix}qfreq` (member=question, score=count),
  `ZINCRBY` + cap top-200 + 30-day TTL refresh on every doc-route question
  (cache hits included).
- `POST /cache/warm` or the conversational `warm_cache` action starts a
  background job (reingest pattern): top-N questions -> skip scoped questions
  (the live pipeline never lets scoped answers into the general cache) ->
  skip fresh hits -> retrieve chunks (refusals are never cached) -> generate
  standalone answers with the economy lane -> store into the semantic cache
  stamped with the current corpus version.
- Guarded by a per-tenant `SET NX EX` lock (touched every step so long runs
  never let a second job slip in); publishes `cache.warm_started` /
  `cache.warm_completed` events.

## Touchpoints

| File | Change |
|---|---|
| `config.py` | `LLM_ECONOMY_*`, `ENABLE_CACHE_WARM`, `CACHE_WARM_TOP_N`, `QFREQ_*` |
| `llm.py` | `lane=` param on `chat`/`stream_chat`; per-lane provider chains |
| `warm.py` (new) | popularity tracking + background warmer |
| `api.py` | chitchat/memory-synthesis on economy; lane counters; `/health`; qfreq hook; `POST /cache/warm` |
| `actions.py` / `router.py` | `warm_cache` handler (non-destructive) + ACTIONS tuple |
| `metrics.py` | `llm_calls_{generation,fast,economy}_total`, `cache_warm_runs_total`, `cache_warm_answers_total` |
| `events.py` | `cache.warm_started` / `cache.warm_completed` |
| `docker-compose.yml` | AMS models -> flash-lite (embeddings untouched) |

(AMS model strings: `gemini/gemini-3.5-flash-lite` - the 2.5 lite tier was
retired for new keys in Aug 2026.)

MCP tools need no changes: they call `api._pipeline` in-process and inherit
the lane routing automatically.

## Tradeoffs

- Warm answers are generated without session/memory context: standalone
  phrasing, fine for cached Q&A.
- Chitchat/synthesis quality rides a small model (worst case an 8B); in
  exchange those replies survive partial outages of either provider.
- AMS has no native model fallback: if Google quota dies, long-term memory
  promotion pauses until reset (chat keeps working). Escape hatch documented
  in docker-compose: flip to `groq/openai/gpt-oss-20b`.
- Pre-existing behaviour left unchanged: fast-lane calls already fall back to
  `gemini-2.5-flash` when Groq rate-limits.
- Free-tier caveat (pre-existing): Google may train on free-tier prompts.

## Verification

`scripts/step12_check_multi_model.py`: economy chain resolution (+ fail-open
when unconfigured), `/health` fields, chitchat bumps the economy counter,
warm run grows the cache with corpus-stamped entries, lock rejects overlap,
routing regression drills still pass.
