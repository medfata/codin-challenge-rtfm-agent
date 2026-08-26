"""Ingestion endpoints: POST /ingest and GET /docs/status."""

from fastapi import APIRouter, Depends, HTTPException

from rtfm_agent.api import state
from rtfm_agent.api.schemas import IngestRequest, IngestResponse
from rtfm_agent.common import events as events_mod
from rtfm_agent.common.paths import resolve_docs_dir
from rtfm_agent.common.tenancy import TenantContext, require_tenant
from rtfm_agent.ingestion import versioning
from rtfm_agent.ingestion.pipeline import run_ingestion

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest | None = None,
           t: TenantContext = Depends(require_tenant)):
    r = state.get_redis()
    path = resolve_docs_dir(t, req.docs_dir if req else None)
    try:
        summary = run_ingestion(r, t, docs_dir=str(path))
    except Exception as exc:
        events_mod.publish(r, t, events_mod.INGEST_FAILED,
                           {"tenant": t.id, "docs_dir": str(path), "error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    return IngestResponse(docs_dir=str(path), **summary)


@router.get("/docs/status")
def docs_status(t: TenantContext = Depends(require_tenant)):
    """Corpus version plus a forced on-disk drift scan for this tenant."""
    from rtfm_agent.common.paths import docs_dir_for

    return versioning.status_report(state.get_redis(), t, docs_dir_for(t))
