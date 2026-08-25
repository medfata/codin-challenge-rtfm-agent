# MCP Integration Plan (Step 10)

Expose the RTFM assistant as an MCP server so external AI agents (Claude Code/Desktop connectors, Cursor, any MCP HTTP client) can use it as a tool. Streamable HTTP, mounted into the existing FastAPI app - one process, one deployment, direct in-process calls to pipeline internals (no self-HTTP).

**Non-goal:** re-proxying long-term memory - the Agent Memory Server already exposes MCP natively on :8002; agents connect to both servers side-by-side.

## Architecture

```
uvicorn :8000 (FastAPI app)
|-- REST routes (/ask, /ingest, /docs/status, ...)      <- unchanged
`-- Mount("/mcp", mcp.streamable_http_app(...))          <- new, ENABLE_MCP=1
      |-- Bearer-token ASGI middleware (optional)
      `-- FastMCP("rtfm") - 5 tools + 1 resource
```

- `stateless_http=True, json_response=True` (SDK-recommended scalable mode)
- Host lifespan enters the MCP app's lifespan via AsyncExitStack (a mounted sub-app's own lifespan never runs otherwise)
- DNS-rebinding guard: TransportSecuritySettings from MCP_ALLOWED_HOSTS (empty -> SDK localhost defaults)

## Tenant resolution (per tool call)

Precedence: X-Tenant-Id header on the /mcp request -> MCP_DEFAULT_TENANT env -> reject with a clear tool error. Same slug regex + allowlist via existing tenancy.normalize_tenant. The header travels through the request context (supported by Claude Code/Cursor HTTP configs).

## Tools

| Tool | Params | Backing | Returns |
|---|---|---|---|
| `ask_question` | question, session_id? | `_pipeline` (routing + RAG + step-9 staleness) in worker thread | `{session_id, answer, citations[], stale, warning, cached, route}` |
| `search_documents` | query, k?=5 | `retrieve()` raw KNN | chunk dicts |
| `list_documents` | - | actions._list_documents logic | text (version + drift notice) |
| `documentation_status` | - | versions.status_report | corpus + drift JSON |
| `service_metrics` | - | metrics.snapshot + cache size | counters |

Resource: `docs://status`. Destructive ops excluded (REST-only). New fail-open record_mcp_call() counter.

## Touchpoints

| File | Change |
|---|---|
| `rtfm_agent/mcp_server.py` (new) | create_mcp_server(get_redis) factory, tools, tenant resolver, bearer-token ASGI middleware |
| `rtfm_agent/api.py` | conditional mount at /mcp, lifespan wiring |
| `rtfm_agent/config.py` | ENABLE_MCP, MCP_DEFAULT_TENANT, MCP_BEARER_TOKEN, MCP_ALLOWED_HOSTS |
| `rtfm_agent/metrics.py` | record_mcp_call() + mcp_calls_total in snapshot |
| `rtfm_agent/actions.py` | metrics report line |
| `pyproject.toml` | official mcp SDK dependency |
| `.env.example` | new keys |
| `ARCHITECTURE.md` | Step 10 section + component-map row |
| `scripts/step10_check_mcp.py` (new) | verification drill |

## Test plan

1. Units: tenant resolution matrix; middleware allow/deny; tool registry names
2. Live via official SDK Client over streamable HTTP: initialize/list_tools; read-only tool round-trips under tenant step10a
3. Fallback drill: no-header client works only with MCP_DEFAULT_TENANT set server-side
4. Isolation: step10a ingested mini-corpus vs step10b empty
5. Bearer drill: token enforced when set, open when unset
6. search_documents/ask_question: SKIP unless embedder/LLM reachable
7. mcp_calls_total increments; key hygiene scan

## Tradeoffs

- HTTP-only v1: stdio-style Claude Desktop setups excluded until a thin stdio wrapper lands (same registry would serve it)
- Bearer token is a shared secret, not per-agent identity
- Stateless mode drops cross-reconnect resumability - fine for request/response tool calls
- All agents share one tenant's quota/cache; destructive actions unreachable via MCP
