"""Crawl job lifecycle: validation, background worker, Redis job records.

Flow: start_job() spawns a worker that walks a website (sitemap.xml
fast-path, then a same-host breadth-first link frontier), extracts readable
markdown, and writes pages to a per-job STAGING directory. Nothing touches
the tenant corpus until a human approves the staged pages (crawler.review).

Job lifecycle lives in Redis (`t:{org}:crawl:{job_id}` + a ZSET index) while
page payloads live on disk; both are swept after crawl.stage_ttl_h hours.
Every failure mode degrades to a recorded failure, never a raised exception
inside the worker thread.
"""

import json
import logging
import shutil
import threading
import time
import uuid
from collections import Counter, deque
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from redis import Redis

from rtfm_agent.common.events import CRAWL_FAILED, CRAWL_STAGED, publish
from rtfm_agent.common.metrics import record_crawl
from rtfm_agent.common.paths import resolve_docs_dir
from rtfm_agent.common.tenancy import TenantContext
from rtfm_agent.config import settings
from rtfm_agent.crawler.fetch import (
    USER_AGENT,
    CrawlError,
    SeedURL,
    assert_public_host,
    extract_content,
    is_asset,
    links_in,
    load_robots,
    load_sitemap,
    normalize_url,
    page_id_for,
    slug_for,
)

logger = logging.getLogger(__name__)


class JobNotFoundError(CrawlError):
    """Unknown or already-expired job id."""


class JobStateError(CrawlError):
    """Operation invalid for the job's current lifecycle state."""


# One crawl at a time per tenant (in-process guard; matches the reingest
# threading model - single-process deployments only).
_ACTIVE: set[str] = set()
_ACTIVE_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Paths / Redis keys
# --------------------------------------------------------------------------


def tenant_web_root(t: TenantContext) -> Path:
    """Approved crawled pages: docs/web/<org>/<host>/<slug>.md."""
    return Path(settings.docs.web_dir) / t.id


def staging_root(t: TenantContext) -> Path:
    """Staged-but-unreviewed jobs live OUTSIDE the approved tree so the
    ingestion loader (which globs docs/web/<org>/ recursively) can never
    pick up unapproved content."""
    return Path(settings.docs.web_dir) / "_staging" / t.id


def job_dir_for(t: TenantContext, job_id: str) -> Path:
    return staging_root(t) / job_id


def _record_key(t: TenantContext, job_id: str) -> str:
    return f"{t.prefix}crawl:{job_id}"


def _index_key(t: TenantContext) -> str:
    return f"{t.prefix}crawl:jobs"


def write_record(r: Redis, t: TenantContext, job_id: str, **fields) -> None:
    payload = {"job_id": job_id, "updated_at": str(time.time()), **fields}
    try:
        pipe = r.pipeline(transaction=False)
        pipe.hset(_record_key(t, job_id), mapping=payload)
        if fields.get("status") == "running":
            pipe.zadd(_index_key(t), {job_id: float(payload.get("created_at",
                                                               time.time()))})
        pipe.execute()
    except Exception as exc:
        logger.warning("crawl record write failed (%s): %s", job_id, exc)


def get_record(r: Redis, t: TenantContext, job_id: str) -> dict | None:
    data = r.hgetall(_record_key(t, job_id))
    if not data:
        return None
    out = {}
    for k, v in data.items():
        key = k.decode() if isinstance(k, bytes) else k
        val = v.decode(errors="replace") if isinstance(v, bytes) else v
        out[key] = val
    return out


def sweep_expired(r: Redis, t: TenantContext) -> int:
    """Drop staging dirs + records for jobs older than stage_ttl_h."""
    cutoff = time.time() - settings.crawl.stage_ttl_h * 3600
    try:
        stale = r.zrangebyscore(_index_key(t), "-inf", cutoff)
    except Exception as exc:
        logger.warning("crawl sweep lookup failed: %s", exc)
        return 0
    removed = 0
    for raw in stale:
        job_id = raw.decode() if isinstance(raw, bytes) else raw
        shutil.rmtree(job_dir_for(t, job_id), ignore_errors=True)
        try:
            pipe = r.pipeline(transaction=False)
            pipe.delete(_record_key(t, job_id))
            pipe.zrem(_index_key(t), job_id)
            pipe.execute()
            removed += 1
        except Exception as exc:
            logger.warning("crawl sweep cleanup failed for %s: %s", job_id, exc)
    return removed


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


def start_job(r: Redis, t: TenantContext, start_url: str,
              max_pages: int | None = None, max_depth: int | None = None,
              path_prefix: str | None = None,
              auto_ingest: bool = False) -> dict:
    """Validate + register a crawl job and launch its worker thread."""
    if not settings.crawl.enabled:
        raise CrawlError("web crawling is disabled (ENABLE_WEB_CRAWL=0)")
    sweep_expired(r, t)

    seed = SeedURL(start_url, path_prefix=path_prefix)
    assert_public_host(seed.host)

    with _ACTIVE_LOCK:
        if t.id in _ACTIVE:
            raise CrawlError(f"a crawl is already running for tenant '{t.id}'")
        _ACTIVE.add(t.id)

    job_id = uuid.uuid4().hex[:12]
    effective_pages = min(max_pages or settings.crawl.max_pages,
                          settings.crawl.hard_page_cap)
    effective_depth = settings.crawl.max_depth if max_depth is None else max_depth

    write_record(r, t, job_id, status="running",
                 seed_url=seed.url,
                 created_at=str(time.time()),
                 max_pages=str(effective_pages), max_depth=str(effective_depth),
                 path_prefix=seed.path_prefix, auto_ingest=str(auto_ingest))

    worker = threading.Thread(
        target=_run_job,
        args=(r, t, job_id, seed, effective_pages, effective_depth, auto_ingest),
        daemon=True, name=f"rtfm-crawl-{t.id}-{job_id}",
    )
    worker.start()
    return {
        "job_id": job_id, "status": "started", "tenant": t.id,
        "seed_url": seed.url, "max_pages": effective_pages,
        "max_depth": effective_depth,
    }


def _run_job(r: Redis, t: TenantContext, job_id: str, seed: SeedURL,
             max_pages: int, max_depth: int, auto_ingest: bool) -> None:
    t0 = time.time()
    jdir = job_dir_for(t, job_id)
    pages_dir = jdir / "pages"
    pages: list[dict] = []
    skipped: Counter = Counter()
    failures = 0
    fetched = 0
    client: httpx.Client | None = None
    try:
        pages_dir.mkdir(parents=True, exist_ok=True)
        client = httpx.Client(
            timeout=settings.crawl.timeout_s, follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        robots = load_robots(client, seed)
        sitemap_budget = max(max_pages * 5, 100)
        sitemap_urls = load_sitemap(client, seed, sitemap_budget)

        queue: deque[tuple[str, int]] = deque([(seed.url, 0)])
        seen = {seed.url}
        for url in sitemap_urls:
            if url not in seen:
                seen.add(url)
                queue.append((url, min(1, max_depth)))

        last_fetch_ts = 0.0
        delay_s = settings.crawl.delay_ms / 1000.0

        while queue and fetched < max_pages:
            url, depth = queue.popleft()
            if not robots.can_fetch(USER_AGENT, url):
                skipped["robots_blocked"] += 1
                continue
            wait = delay_s - (time.time() - last_fetch_ts)
            if wait > 0:
                time.sleep(wait)

            try:
                last_fetch_ts = time.time()
                resp = client.get(url)
                fetched += 1
                final_url = normalize_url(str(resp.url)) or url
                if resp.status_code != 200:
                    skipped[f"http_{resp.status_code}"] += 1
                    continue
                ctype = resp.headers.get("content-type", "").split(";")[0].strip()
                if ctype not in ("text/html", "application/xhtml+xml"):
                    skipped["content_type"] += 1
                    continue

                body = resp.content[:settings.crawl.max_bytes]
                page_html = body.decode(
                    resp.encoding or "utf-8", errors="replace")
                title, text = extract_content(page_html, final_url)
                if len(text) < settings.crawl.min_text_chars:
                    skipped["thin_content"] += 1
                    continue

                pid = page_id_for(final_url)
                host_dir, rel_path = slug_for(final_url, pid)
                staged_name = f"pages/{pid}.md"
                (jdir / staged_name).write_text(
                    f"== {title}\n\nSource: {final_url}\n\n{text}\n",
                    encoding="utf-8",
                )
                pages.append({
                    "page_id": pid, "url": final_url, "title": title,
                    "host": host_dir, "rel_path": rel_path,
                    "file": staged_name, "chars": len(text), "status": "ok",
                })

                if depth < max_depth:
                    for href in links_in(page_html):
                        link = normalize_url(href, base=final_url)
                        if (not link or link in seen or not seed.allows(link)
                                or is_asset(urlsplit(link).path)):
                            continue
                        seen.add(link)
                        queue.append((link, depth + 1))
            except httpx.HTTPError as exc:
                failures += 1
                skipped["network_errors"] += 1
                logger.warning("crawl fetch failed for %s: %s", url, exc)
            except Exception as exc:  # extraction/parsing bugs never kill the job
                failures += 1
                skipped["processing_errors"] += 1
                logger.warning("crawl processing failed for %s: %s", url, exc)

        duration = round(time.time() - t0, 2)
        summary = {
            "seed_url": seed.url, "pages_found": len(seen),
            "pages_fetched": fetched, "pages_staged": len(pages),
            "skipped": dict(skipped), "failures": failures,
            "duration_s": duration,
        }
        status = "awaiting_review" if pages else "failed"
        manifest = {
            "job_id": job_id, "tenant": t.id, "status": status,
            "seed_url": seed.url, "created_at": time.time(),
            "summary": summary, "pages": pages,
        }
        (jdir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_record(r, t, job_id, status=status,
                     pages_total=str(len(seen)), pages_staged=str(len(pages)),
                     failures=str(failures),
                     summary_json=json.dumps(summary, ensure_ascii=False))
        record_crawl(r, t, pages_fetched=fetched, failures=failures)
        publish(
            r, t,
            CRAWL_STAGED if pages else CRAWL_FAILED,
            {"job_id": job_id, "tenant": t.id, **summary},
        )
    except Exception as exc:
        logger.exception("crawl job %s crashed", job_id)
        write_record(r, t, job_id, status="failed", error=str(exc)[:500],
                     finished_at=str(time.time()))
        record_crawl(r, t, pages_fetched=fetched, failures=failures + 1)
        publish(r, t, CRAWL_FAILED,
                {"job_id": job_id, "tenant": t.id, "error": str(exc)})
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE.discard(t.id)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    if auto_ingest and pages:
        try:
            # Lazy imports: review pulls the manifest machinery and ingest
            # pulls the embedder; keep this module importable without them.
            from rtfm_agent.crawler.review import approve_job, mark_ingested
            from rtfm_agent.ingestion.pipeline import run_ingestion

            approve_job(r, t, job_id, exclude_ids=None)
            docs_dir = str(resolve_docs_dir(t))
            run_ingestion(r, t, docs_dir=docs_dir)
            mark_ingested(r, t, job_id)
        except Exception as exc:
            logger.error("auto-ingest after crawl %s failed: %s", job_id, exc)
            write_record(r, t, job_id, status="failed",
                         error=str(exc)[:500], finished_at=str(time.time()))
