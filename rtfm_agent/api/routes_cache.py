"""Semantic-cache management endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from rtfm_agent.api import state
from rtfm_agent.common.tenancy import TenantContext, require_tenant
from rtfm_agent.config import settings
from rtfm_agent.retrieval import cache as cache_mod
from rtfm_agent.routing import warm as warm_mod

router = APIRouter()


@router.post("/cache/flush")
def flush_cache(t: TenantContext = Depends(require_tenant)):
    removed = cache_mod.flush(state.get_redis(), t)
    return {"removed": removed, "tenant": t.id}


@router.post("/cache/warm")
def warm_cache(t: TenantContext = Depends(require_tenant)):
    """Pre-answer this tenant's most-asked questions with the economy lane.

    Runs in a background thread and acknowledges immediately; a second run
    while one is active is dropped (logged + cache.warm_completed event).
    """
    if not settings.warm.enabled:
        raise HTTPException(
            status_code=503, detail="cache warming is disabled (ENABLE_CACHE_WARM=0)"
        )
    warm_mod.start_background(state.get_redis(), t, settings.warm.top_n)
    return {"started": True, "tenant": t.id, "top_n": settings.warm.top_n}
