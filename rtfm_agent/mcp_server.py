"""Step 10: MCP exposure - the assistant as a tool server for other agents.

Mounts a streamable-HTTP MCP server at `/mcp` inside the existing FastAPI
app (see api.lifespan for the session-manager wiring). Tools call the same
internals the REST API uses - the RAG pipeline, actions, versioning and
metrics modules - directly in-process; there is no self-HTTP hop.

Identity reuses the deployment's trusted-header model: tools resolve their
tenant from the X-Tenant-Id header carried on the /mcp request (supported by
Claude Code / Cursor / Claude Desktop remote connector configs), falling
back to MCP_DEFAULT_TENANT when absent. An optional shared bearer secret can
gate the whole endpoint via MCP_BEARER_TOKEN.

Destructive operations are deliberately not exposed: flushing caches or
re-ingesting remains a REST/operator decision.
"""

import functools
import logging
import uuid
from typing import Callable, NamedTuple

from anyio import to_thread
from fastapi import HTTPException
from redis import Redis
from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings

from rtfm_agent.config import (
    MCP_ALLOWED_HOSTS,
    MCP_BEARER_TOKEN,
    MCP_DEFAULT_TENANT,
)
from rtfm_agent.tenancy import TenantContext, normalize_tenant

logger = logging.getLogger(__name__)


class McpMount(NamedTuple):
    """Everything api.py needs to host the MCP server."""

    server: MCPServer          # registry handle (introspection/tests)
    app: object                # ASGI app to mount at /mcp (maybe wrapped)
    lifespan_app: object       # the Starlette whose lifespan must be entered


class BearerTokenMiddleware:
    """Minimal ASGI middleware: require `Authorization: Bearer <token>`."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            supplied = Headers(scope=scope).get("authorization", "")
            if supplied != f"Bearer {self.token}":
                response = JSONResponse(
                    {"error": "unauthorized: missing or invalid bearer token"},
                    status_code=401,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def resolve_tenant(headers) -> TenantContext:
    """Tenant for one MCP call: X-Tenant-Id header, else MCP_DEFAULT_TENANT.

    Raises ToolError so agents get an actionable message instead of a silent
    default tenant. Accepts any mapping (Starlette Headers are provided by
    the SDK's request context).
    """
    raw = None
    mapping = headers or {}
    try:
        items = mapping.items()
    except AttributeError:
        items = []
    for key, value in items:
        if str(key).lower() == "x-tenant-id":
            raw = value
            break
    ctx = normalize_tenant(raw)
    if ctx is None and MCP_DEFAULT_TENANT:
        ctx = normalize_tenant(MCP_DEFAULT_TENANT)
    if ctx is None:
        raise ToolError(
            "missing or invalid X-Tenant-Id header "
            "(and no MCP_DEFAULT_TENANT fallback configured): "
            "expected slug [a-z0-9][a-z0-9-_]{0,62}"
        )
    return ctx


def _docs_dir_for(t: TenantContext) -> str:
    """Mirror api._resolve_docs_dir precedence minus the per-request override."""
    from pathlib import Path

    from rtfm_agent.config import DOCS_DIR

    project_root = Path(__file__).resolve().parent.parent
    team_dir = project_root / "docs" / t.id
    if team_dir.is_dir():
        return str(team_dir)
    return str(Path(DOCS_DIR))


def create_mcp_server(get_redis: Callable[[], Redis]) -> McpMount:
    """Build the MCP server + mountable ASGI app.

    `get_redis` injects the API's Redis connection (avoids an import cycle
    with api.py); the RAG pipeline still runs through api._pipeline, which
    uses that same connection.
    """
    mcp = MCPServer(
        name="rtfm",
        title="RTFM For Me Agent",
        description=(
            "Documentation assistant: grounded Q&A over the tenant's indexed "
            "corpus with source citations and staleness warnings."
        ),
        version="0.10.0",
    )

    def _count(t: TenantContext) -> None:
        from rtfm_agent import metrics as metrics_mod

        metrics_mod.record_mcp_call(get_redis(), t)

    @mcp.tool(
        title="Ask the documentation assistant",
        description=(
            "Ask a question about the indexed documentation. Returns a "
            "grounded answer with source citations. Handles follow-ups when "
            "you pass back the session_id from a previous reply; each reply "
            "includes one. The 'stale'/'warning' fields flag answers built "
            "on outdated documentation."
        ),
    )
    async def ask_question(question: str, session_id: str | None = None,
                           ctx: Context = None) -> dict:
        t = resolve_tenant(ctx.headers if ctx else None)
        _count(t)
        from rtfm_agent.api import _pipeline

        # The pipeline assumes the caller generated the id (REST does); a
        # blank id would funnel every stateless agent call into one session.
        sid = (session_id or "").strip() or uuid.uuid4().hex
        try:
            result = await to_thread.run_sync(
                functools.partial(_pipeline, question, sid, t)
            )
        except HTTPException as exc:
            raise ToolError(str(exc.detail))
        except Exception as exc:
            raise ToolError(f"ask_question failed: {exc}")
        return {
            "session_id": sid,
            "answer": result["answer"],
            "citations": [c.model_dump() for c in result.get("citations", [])],
            "sources_consulted": result.get("sources_consulted", 0),
            "cached": result.get("cached", False),
            "stale": result.get("stale", False),
            "warning": result.get("warning"),
            "route": result.get("route", "doc"),
        }

    @mcp.tool(
        title="Search document chunks",
        description=(
            "Raw semantic search over the indexed documentation: returns the "
            "top-k chunks (text, source file, heading, position, distance "
            "score) without generating an answer. Use ask_question instead "
            "when you want a synthesized, cited answer. You judge chunk "
            "relevance yourself from 'score' (lower cosine distance = closer)."
        ),
    )
    async def search_documents(query: str, k: int = 5, ctx: Context = None) -> dict:
        t = resolve_tenant(ctx.headers if ctx else None)
        _count(t)

        def _search():
            import numpy as np

            from rtfm_agent.retrieval import get_embedder, retrieve

            r = get_redis()
            qvec = get_embedder().embed([query])[0].astype(np.float32).tobytes()
            return retrieve(query, r, t, k=max(1, min(int(k), 20)), qvec=qvec)

        try:
            chunks = await to_thread.run_sync(_search)
        except Exception as exc:
            raise ToolError(f"search_documents failed: {exc}")
        return {"query": query, "results": chunks}

    @mcp.tool(
        title="List indexed documents",
        description=(
            "List this tenant's indexed documentation files with chunk "
            "counts, the current corpus version, and a note when files have "
            "changed on disk since the last ingestion."
        ),
    )
    async def list_documents(ctx: Context = None) -> str:
        t = resolve_tenant(ctx.headers if ctx else None)
        _count(t)
        from rtfm_agent.actions import _list_documents

        try:
            return await to_thread.run_sync(
                functools.partial(_list_documents, get_redis(), t, "")
            )
        except Exception as exc:
            raise ToolError(f"list_documents failed: {exc}")

    def _status(t: TenantContext) -> dict:
        from rtfm_agent import versions as versions_mod

        return versions_mod.status_report(get_redis(), t, _docs_dir_for(t))

    @mcp.tool(
        title="Documentation status",
        description=(
            "Versioning state of the documentation corpus: current corpus "
            "version/digest/ingest time plus which files changed, appeared, "
            "or disappeared on disk since the last ingestion. Check this to "
            "decide whether answers may be outdated."
        ),
    )
    async def documentation_status(ctx: Context = None) -> dict:
        t = resolve_tenant(ctx.headers if ctx else None)
        _count(t)
        try:
            return await to_thread.run_sync(functools.partial(_status, t))
        except Exception as exc:
            raise ToolError(f"documentation_status failed: {exc}")

    @mcp.tool(
        title="Service metrics",
        description=(
            "Operational stats for this tenant: request/error counts, "
            "semantic-cache hit rate and latency, token usage and estimated "
            "cost, route mix, MCP calls, and stale-answer count."
        ),
    )
    async def service_metrics(ctx: Context = None) -> dict:
        t = resolve_tenant(ctx.headers if ctx else None)
        _count(t)

        def _metrics():
            from rtfm_agent import cache as cache_mod
            from rtfm_agent import metrics as metrics_mod

            r = get_redis()
            snap = metrics_mod.snapshot(r, t)
            snap["cache_size"] = cache_mod.count_entries(r, t)
            snap["tenant"] = t.id
            return snap

        try:
            return await to_thread.run_sync(_metrics)
        except Exception as exc:
            raise ToolError(f"service_metrics failed: {exc}")

    # Templated URI because the SDK forbids Context injection on static
    # resources - the tenant rides in the URI and is validated like a header.
    def read_docs_status(tenant: str) -> dict:
        from mcp.server.mcpserver.exceptions import ResourceError

        t = normalize_tenant(tenant)
        if t is None:
            raise ResourceError(
                f"invalid tenant slug {tenant!r} "
                "(expected [a-z0-9][a-z0-9-_]{0,62}, within the allowlist)"
            )
        _count(t)
        try:
            return _status(t)
        except Exception as exc:
            raise ResourceError(f"read docs://status failed: {exc}")

    mcp.resource(
        "docs://status/{tenant}",
        name="Documentation status",
        description="Corpus version and drift report for {tenant} (same "
                    "payload as the documentation_status tool).",
        mime_type="application/json",
    )(read_docs_status)

    security = (
        TransportSecuritySettings(allowed_hosts=MCP_ALLOWED_HOSTS, allowed_origins=[])
        if MCP_ALLOWED_HOSTS else None
    )
    # Default inner path keeps the endpoint at /mcp: api.py mounts this app
    # at root AFTER its REST routes, so Starlette order gives REST priority
    # and /mcp falls through here (no Mount-prefix redirects).
    starlette_app = mcp.streamable_http_app(
        json_response=True,
        stateless_http=True,
        transport_security=security,
    )
    asgi_app = (
        BearerTokenMiddleware(starlette_app, MCP_BEARER_TOKEN)
        if MCP_BEARER_TOKEN else starlette_app
    )
    return McpMount(server=mcp, app=asgi_app, lifespan_app=starlette_app)
