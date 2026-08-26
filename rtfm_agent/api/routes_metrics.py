"""Observability endpoints: GET /metrics and GET /health."""

from fastapi import APIRouter, Depends, HTTPException

from rtfm_agent.api import state, schemas
from rtfm_agent.common import sessions as sessions_mod
from rtfm_agent.common.metrics import snapshot
from rtfm_agent.common.tenancy import TenantContext, require_tenant
from rtfm_agent.config import settings
from rtfm_agent.retrieval import cache as cache_mod
from rtfm_agent.routing.memory import is_healthy as memory_healthy

router = APIRouter()


@router.get("/metrics", response_model=schemas.MetricsResponse)
def get_metrics(t: TenantContext = Depends(require_tenant)):
    r = state.get_redis()
    snap = snapshot(r, t)
    snap["cache_size"] = cache_mod.count_entries(r, t)
    return schemas.MetricsResponse(**snap)


@router.get("/health")
def health():
    r = state.get_redis()
    try:
        pong = r.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis unreachable: {exc}")
    llm = settings.llm
    return {
        "status": "ok" if pong else "degraded",
        "redis": "up" if pong else "down",
        "llm_model": llm.model,
        "llm_fast_model": llm.fast_model,
        "llm_economy_model": llm.economy_model,
        "llm_economy_fallback_model": llm.economy_fallback_model,
        "llm_economy_configured": bool(llm.economy_api_key),
        "llm_configured": bool(llm.api_key),
        "fallback_llm_configured": bool(llm.fallback_api_key),
        "cache_threshold": settings.cache.threshold,
        "query_rewrite": settings.sessions.enable_query_rewrite,
        "routing": settings.routing.enabled,
        "cache_warm": settings.warm.enabled,
        "session_ttl_s": settings.sessions.ttl_seconds,
        "memory_server": "up" if memory_healthy() else "down",
    }
