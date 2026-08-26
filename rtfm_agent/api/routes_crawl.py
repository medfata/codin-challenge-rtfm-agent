"""Web-crawl endpoints: discover, stage for review, approve into the corpus."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from rtfm_agent.api import state
from rtfm_agent.api.schemas import CrawlApproveRequest, CrawlRequest
from rtfm_agent.common.paths import resolve_docs_dir
from rtfm_agent.common.tenancy import TenantContext, require_tenant
from rtfm_agent.config import settings
from rtfm_agent.crawler import (
    CrawlError,
    JobNotFoundError,
    JobStateError,
    approve_job,
    discard_job,
    get_job,
    list_jobs,
    mark_ingested,
    start_job,
)
from rtfm_agent.ingestion.pipeline import run_ingestion

router = APIRouter()


def _crawl_error(exc: Exception) -> HTTPException:
    if isinstance(exc, JobNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, JobStateError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))


@router.post("/crawl", status_code=202)
def start_crawl(req: CrawlRequest,
                t: TenantContext = Depends(require_tenant)):
    """Crawl a documentation site and stage extracted pages for review.

    Nothing is indexed until POST /crawl/jobs/{id}/approve (or auto_ingest).
    """
    if not settings.crawl.enabled:
        raise HTTPException(
            status_code=503, detail="web crawling is disabled (ENABLE_WEB_CRAWL=0)"
        )
    try:
        job = start_job(
            state.get_redis(), t, req.start_url, max_pages=req.max_pages,
            max_depth=req.max_depth, path_prefix=req.path_prefix,
            auto_ingest=req.auto_ingest,
        )
    except CrawlError as exc:
        raise _crawl_error(exc)
    return {"job_id": job["job_id"], "status": job["status"], "tenant": t.id,
            "seed_url": job["seed_url"], "max_pages": job["max_pages"],
            "max_depth": job["max_depth"],
            "review_url": "/crawl/review" if not req.auto_ingest else None}


@router.get("/crawl/jobs")
def list_crawl_jobs(t: TenantContext = Depends(require_tenant)):
    return {"tenant": t.id, "jobs": list_jobs(state.get_redis(), t)}


@router.get("/crawl/review", include_in_schema=False)
def crawl_review_page():
    """Serve the staged-crawl review UI same-origin (no CORS setup needed)."""
    page = Path(__file__).resolve().parents[2] / "static" / "crawl_review.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="review page not found")
    return FileResponse(page, media_type="text/html")


@router.get("/crawl/jobs/{job_id}")
def get_crawl_job(job_id: str, t: TenantContext = Depends(require_tenant)):
    try:
        return get_job(state.get_redis(), t, job_id)
    except CrawlError as exc:
        raise _crawl_error(exc)


@router.get("/crawl/jobs/{job_id}/pages/{page_id}")
def preview_crawl_page(job_id: str, page_id: str,
                       t: TenantContext = Depends(require_tenant)):
    """Full extracted text of one staged page - the human verification step."""
    from rtfm_agent.crawler import read_page

    try:
        return read_page(t, job_id, page_id)
    except CrawlError as exc:
        raise _crawl_error(exc)


@router.post("/crawl/jobs/{job_id}/approve")
def approve_crawl_job(job_id: str, req: CrawlApproveRequest | None = None,
                      t: TenantContext = Depends(require_tenant)):
    """Merge kept pages into the corpus and re-ingest this tenant's docs."""
    r = state.get_redis()
    try:
        approved = approve_job(r, t, job_id,
                               exclude_ids=req.exclude if req else [])
    except CrawlError as exc:
        raise _crawl_error(exc)
    summary = run_ingestion(r, t, docs_dir=str(resolve_docs_dir(t)))
    mark_ingested(r, t, job_id)
    return {"job_id": job_id, "tenant": t.id, **approved, "ingestion": summary}


@router.delete("/crawl/jobs/{job_id}")
def discard_crawl_job(job_id: str, t: TenantContext = Depends(require_tenant)):
    try:
        result = discard_job(state.get_redis(), t, job_id)
    except CrawlError as exc:
        raise _crawl_error(exc)
    return {**result, "tenant": t.id}
