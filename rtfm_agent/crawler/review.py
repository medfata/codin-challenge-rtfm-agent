"""Crawl review operations: list jobs, inspect staged pages, approve them
into the tenant corpus, or discard them.

The review gate is the safety boundary between automated crawling and the
indexed corpus: nothing is visible to the RAG pipeline until approve_job()
copies kept pages into docs/web/<org>/<host>/ and ingestion runs.
"""

import json
import logging
import shutil
import time

from redis import Redis

from rtfm_agent.common.metrics import record_crawl
from rtfm_agent.common.tenancy import TenantContext
from rtfm_agent.crawler.job import (
    JobNotFoundError,
    JobStateError,
    get_record,
    job_dir_for,
    tenant_web_root,
    write_record,
)

logger = logging.getLogger(__name__)


def list_jobs(r: Redis, t: TenantContext, limit: int = 20) -> list[dict]:
    try:
        ids = r.zrevrange(f"{t.prefix}crawl:jobs", 0, max(limit - 1, 0))
    except Exception as exc:
        logger.warning("crawl job listing failed: %s", exc)
        return []
    jobs = []
    for raw in ids:
        job_id = raw.decode() if isinstance(raw, bytes) else raw
        record = get_record(r, t, job_id)
        if record:
            jobs.append({"job_id": job_id, **record})
    return jobs


def get_job(r: Redis, t: TenantContext, job_id: str) -> dict:
    record = get_record(r, t, job_id)
    if record is None:
        raise JobNotFoundError(f"unknown crawl job '{job_id}'")
    job = {"job_id": job_id, **record}
    if job.get("summary_json"):
        try:
            job["summary"] = json.loads(job["summary_json"])
        except ValueError:
            pass
    # Staging details only matter while a job awaits review; once approved,
    # the Redis record is the source of truth (a surviving manifest.json -
    # e.g. a Windows lock lost the rmtree race - must never override it).
    if job.get("status") == "awaiting_review":
        manifest = _load_manifest_if_exists(t, job_id)
        if manifest is not None:
            job["summary"] = manifest.get("summary", job.get("summary"))
            job["pages"] = [
                {k: p[k] for k in ("page_id", "url", "title", "chars", "status")}
                for p in manifest.get("pages", [])
            ]
    return job


def read_page(t: TenantContext, job_id: str, page_id: str) -> dict:
    """Full extracted text of one staged page (for human verification)."""
    manifest = _load_manifest_if_exists(t, job_id)
    if manifest is None:
        raise JobNotFoundError(
            f"job '{job_id}' has no staged pages (already reviewed/expired?)")
    for page in manifest.get("pages", []):
        if page["page_id"] == page_id:
            path = job_dir_for(t, job_id) / page["file"]
            if not path.is_file():
                raise JobNotFoundError(f"staged file for page '{page_id}' is missing")
            text = path.read_text(encoding="utf-8", errors="replace")
            return {k: page[k] for k in ("page_id", "url", "title", "chars")} | {
                "text": text,
            }
    raise JobNotFoundError(f"page '{page_id}' is not part of job '{job_id}'")


def _load_manifest_if_exists(t: TenantContext, job_id: str) -> dict | None:
    path = job_dir_for(t, job_id) / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("manifest read failed for job %s: %s", job_id, exc)
        return None


def approve_job(r: Redis, t: TenantContext, job_id: str,
                exclude_ids: list[str] | None = None) -> dict:
    """Merge kept staged pages into the tenant's approved web corpus.

    Files land at docs/web/<org>/<host>/<slug>-<pid>.md. For every host this
    crawl covered, previously-approved files NOT among the kept set are
    deleted (vanished pages and explicitly excluded ones), so a re-crawl is
    also the removal mechanism. The caller then triggers run_ingestion().
    """
    manifest = _load_manifest_if_exists(t, job_id)
    if manifest is None:
        raise JobNotFoundError(f"unknown or expired crawl job '{job_id}'")
    if manifest.get("status") != "awaiting_review":
        raise JobStateError(
            f"job '{job_id}' is '{manifest.get('status')}', not awaiting review")

    exclude = set(exclude_ids or [])
    web_root = tenant_web_root(t)
    jdir = job_dir_for(t, job_id)
    kept_by_host: dict[str, set[str]] = {}
    approved: list[str] = []

    for page in manifest.get("pages", []):
        if page.get("status") != "ok" or page["page_id"] in exclude:
            continue
        src = jdir / page["file"]
        dest = web_root / page["rel_path"]
        if not src.is_file():
            logger.warning("staged file vanished for %s", page["url"])
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        kept_by_host.setdefault(page["host"], set()).add(page["rel_path"])
        approved.append(page["rel_path"])

    removed: list[str] = []
    for host, kept in kept_by_host.items():
        host_dir = web_root / host
        if not host_dir.is_dir():
            continue
        for existing in sorted(host_dir.rglob("*.md")):
            rel = existing.relative_to(web_root).as_posix()
            if rel not in kept:
                existing.unlink()
                removed.append(rel)
        for sub in sorted((p for p in host_dir.rglob("*") if p.is_dir()),
                          reverse=True):
            try:
                sub.rmdir()
            except OSError:
                pass
        try:
            host_dir.rmdir()
        except OSError:
            pass  # still has kept files

    shutil.rmtree(jdir, ignore_errors=True)
    if jdir.exists():  # Windows: a concurrent reader's lock can beat the unlink
        time.sleep(0.2)
        shutil.rmtree(jdir, ignore_errors=True)
    if jdir.exists():
        logger.warning("staging dir %s could not be fully removed", jdir)
    write_record(r, t, job_id, status="approved", finished_at=str(time.time()),
                 approved=str(len(approved)), removed_files=str(len(removed)))
    return {
        "approved": len(approved),
        "excluded": len([p for p in exclude]),
        "removed_files": len(removed),
        "removed_sample": removed[:20],
        "files": approved,
    }


def mark_ingested(r: Redis, t: TenantContext, job_id: str) -> None:
    """Flip an approved job to ingested once its re-ingestion has completed."""
    write_record(r, t, job_id, status="ingested")


def discard_job(r: Redis, t: TenantContext, job_id: str) -> dict:
    """Throw a staged job away without touching anything approved."""
    record = get_record(r, t, job_id)
    if record is None:
        raise JobNotFoundError(f"unknown crawl job '{job_id}'")
    if record.get("status") not in ("awaiting_review", "failed"):
        raise JobStateError(
            f"job '{job_id}' is '{record.get('status')}' and cannot be discarded")
    shutil.rmtree(job_dir_for(t, job_id), ignore_errors=True)
    write_record(r, t, job_id, status="discarded", finished_at=str(time.time()))
    record_crawl(r, t, discarded=True)
    return {"job_id": job_id, "status": "discarded"}
