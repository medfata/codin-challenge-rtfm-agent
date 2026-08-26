"""FastAPI application factory: lifespan, CORS, routers, and the MCP mount."""

import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from rtfm_agent.api import state
from rtfm_agent.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rtfm.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    r = Redis.from_url(settings.redis.url, decode_responses=False)
    r.ping()
    # Shared async pool for SSE subscribers: each /events/stream client
    # borrows a connection per XREAD await instead of owning one, so
    # hundreds of streams cost no extra sockets or worker threads.
    # socket_timeout MUST exceed the XREAD block window - redis-py 8.x async
    # otherwise falls back to the connect timeout for reads (5s), turning
    # every idle subscriber into a disconnect/reconnect churn loop.
    ar = AsyncRedis.from_url(
        settings.redis.url, decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=settings.events.heartbeat_s + 5,
    )
    await ar.ping()
    state.set_runtime(r, ar)
    # Cache/doc indexes are per-tenant and self-heal lazily on first use.
    # The MCP mount's own lifespan never runs (Starlette drops sub-app
    # lifespans on Mount), so the session manager is entered here.
    try:
        async with contextlib.AsyncExitStack() as stack:
            if settings.mcp.enabled and app.state.mcp_mount is not None:
                mount = app.state.mcp_mount
                await stack.enter_async_context(
                    mount.lifespan_app.router.lifespan_context(mount.lifespan_app)
                )
            yield
    finally:
        await state.aclose_runtime()


app = FastAPI(title="RTFM For Me Agent", version="0.7.1", lifespan=lifespan)

# Cross-origin browser access for the SSE feed (and GETs generally). Scoped
# to read-only methods on purpose; tighten events.cors_origins in production.
if settings.events.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.events.cors_origins,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["Last-Event-ID", "X-Tenant-Id"],
    )

# Built here but mounted at the END of this module: the MCP app sits at root
# serving /mcp internally, and Starlette route order means it must be
# appended after every REST route.
_mcp_mount = None
if settings.mcp.enabled:
    from rtfm_agent.mcp_server import create_mcp_server

    _mcp_mount = create_mcp_server(state.get_redis)

app.state.mcp_mount = _mcp_mount

# ---------------------------------------------------------------------------
# Routers (imported here so every module can rely on runtime state existing)
# ---------------------------------------------------------------------------

from rtfm_agent.api.routes_ask import router as ask_router  # noqa: E402
from rtfm_agent.api.routes_cache import router as cache_router  # noqa: E402
from rtfm_agent.api.routes_crawl import router as crawl_router  # noqa: E402
from rtfm_agent.api.routes_events import router as events_router  # noqa: E402
from rtfm_agent.api.routes_ingest import router as ingest_router  # noqa: E402
from rtfm_agent.api.routes_metrics import router as metrics_router  # noqa: E402
from rtfm_agent.api.routes_sessions import router as sessions_router  # noqa: E402

app.include_router(ask_router)
app.include_router(cache_router)
app.include_router(crawl_router)
app.include_router(events_router)
app.include_router(ingest_router)
app.include_router(metrics_router)
app.include_router(sessions_router)


# Must come after every REST route: a root Mount swallows all later routes.
if settings.mcp.enabled and _mcp_mount is not None:
    app.mount("/", _mcp_mount.app)
