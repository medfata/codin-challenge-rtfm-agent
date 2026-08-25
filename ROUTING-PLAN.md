# Semantic Routing Plan (post-challenge feature)

Classify every incoming message before processing, then route it to the right handler.

## Routes

| Route | Trigger | Behavior |
|---|---|---|
| `chitchat` | greetings, thanks, small talk | Canned reply if pattern matches (zero LLM), else tiny flash reply. Skips retrieval/cache/RAG. Turn kept in session history only |
| `docs` | Git/documentation questions | Existing full pipeline (default) |
| `action` | operational requests | Regex-whitelisted ops only: re-ingest docs (background + ack), flush cache, reset session, forget me |
| `memory` | "what do you know about me?" | Recall top-N long-term memories -> synthesis reply, no doc retrieval |

## Key design decision: ONE combined "intent" call

Instead of separate classify + rewrite calls, one structured flash-model call returns:

```json
{
  "route": "docs | chitchat | action | memory",
  "action": null | "reingest | flush_cache | reset_session | forget_me",
  "search_query": "...",
  "source": null
}
```

Per-message LLM budget: intent (1) + handler (0-1) — same as today for follow-ups.
Semantic-cache hits skip their handler entirely.

Safety: actions require regex-whitelist match on the raw question; the LLM hint only picks which pattern set. Unmatched action falls back to `docs`. Fail-open everywhere: malformed JSON / LLM error / unknown route -> `docs`.

## Flow

```
embed -> intent call -> branch:
  chitchat: canned pattern ? canned : micro-reply
  action:   whitelist match -> execute -> result as answer
  memory:   recall top-N -> synthesis reply
  docs:     existing pipeline (scope/cache/RAG unchanged)
```

## Touchpoints

| File | Change |
|---|---|
| `prompts.py` | `INTENT_SYSTEM` (JSON contract), `CHITCHAT_SYSTEM`, memory-synthesis prompt |
| `rtfm_agent/router.py` (new) | Intent call + defensive parsing + action whitelist matching |
| `api.py` | Branch after intent; `route` field in `/ask`; `event: route` in SSE |
| `metrics.py` | Per-route counters (`route_docs_total`, ...) |
| `config.py` | `ENABLE_ROUTING=1` |

## Test plan

1. Route accuracy over sample phrases per class
2. All 4 action drills with side-effect verification
3. Chitchat latency/cost check (no citations, fast)
4. Memory route surfaces stored facts
5. Step 6 regression: everything still routes `docs`
6. Fail-open drill (malformed classifier output)
7. `/metrics` route counters populated
