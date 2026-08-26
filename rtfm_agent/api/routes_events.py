"""Real-time event endpoints: SSE stream + the demo page."""

import json

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path

from rtfm_agent.api import state
from rtfm_agent.common import events as events_mod
from rtfm_agent.common.tenancy import TenantContext, resolve_tenant
from rtfm_agent.config import settings

router = APIRouter()


def _event_tenant(x_tenant_id: str = Header(default=""),
                  tenant: str | None = Query(default=None)) -> TenantContext:
    """Tenant resolution for /events/stream.

    Browsers' native EventSource cannot set custom headers, so ?tenant= is
    accepted as an equivalent trusted identity (same validation, same trust
    level as the X-Tenant-Id header). Sending both with different values is
    rejected rather than silently picking one.
    """
    if tenant and x_tenant_id.strip() and tenant.strip().lower() != x_tenant_id.strip().lower():
        raise HTTPException(
            status_code=422,
            detail="conflicting tenant identity: ?tenant= and X-Tenant-Id disagree",
        )
    return resolve_tenant(tenant or x_tenant_id)


@router.get("/events/stream")
async def events_stream(
    backlog: int = Query(default=0, ge=0, le=100),
    tenant: str | None = Query(default=None),
    last_event_id: str = Header(default=""),
    t: TenantContext = Depends(_event_tenant),
):
    """SSE feed of this tenant's real-time events.

    Emits ingest.started/completed/failed and memory.turn_stored frames.
    Reconnects resume from the browser's Last-Event-ID header; ?backlog=N
    replays the N most recent entries on a fresh connect. Heartbeat comments
    keep proxies from idling out.

    The read loop is fully async on the shared Redis pool - subscribers
    cost no worker threads, so the sync endpoint pool stays free.
    """
    if not settings.events.enabled:
        raise HTTPException(
            status_code=503, detail="event streaming is disabled (ENABLE_EVENTS=0)"
        )
    ar = state.get_async_redis()

    # Probe eagerly (shared async pool) so a dead Redis is a clean 503
    # instead of a stream that only ever emits heartbeats.
    try:
        await ar.ping()
        cursor = await events_mod.aresolve_start(
            ar, t, events_mod.normalize_last_id(last_event_id), backlog
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis unreachable: {exc}")

    async def generate():
        yield "retry: 3000\n\n"
        yield f": connected tenant={t.id}\n\n"
        async for item in events_mod.aiter_events(ar, t, cursor):
            if item is None:
                yield ": ping\n\n"
                continue
            entry_id, etype, data = item
            payload = json.dumps(data, ensure_ascii=False)
            yield f"id: {entry_id}\nevent: {etype}\ndata: {payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/demo/events", include_in_schema=False)
def events_demo_page():
    """Serve the event-feed demo page same-origin (no CORS setup needed)."""
    page = Path(__file__).resolve().parents[2] / "static" / "events_demo.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="demo page not found")
    return FileResponse(page, media_type="text/html")
