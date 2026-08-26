"""Step 13: web crawl - discover, fetch, extract and STAGE documentation pages.

Flow: POST /crawl -> start_job() spawns a background worker that walks a
website (sitemap.xml fast-path, then a same-host breadth-first link frontier),
extracts readable markdown via trafilatura, and writes pages to a per-job
STAGING directory. Nothing touches the tenant corpus until a human inspects
the staged pages and approves them (approve_job copies kept pages into
docs/web/<org>/<host>/ and the API then triggers a normal ingestion).

Safety rails:
  * http/https only, credentials-in-URL rejected
  * SSRF guard: DNS-resolve the host and reject private/loopback/link-local/
    reserved targets (CRAWL_ALLOW_PRIVATE_HOSTS=1 lifts this for tests ONLY)
  * robots.txt honoured (per-origin cache); polite delay between requests
  * response-size cap, content-type check, page/depth caps

Job lifecycle lives in Redis (`t:{org}:crawl:{job_id}` + a ZSET index) while
page payloads live on disk; both are swept after CRAWL_STAGE_TTL_H hours.
Every failure mode degrades to a recorded failure, never a raised exception
inside the worker thread.
"""

import hashlib
import html as html_mod
import ipaddress
import json
import logging
import re
import shutil
import socket
import threading
import time
import uuid
from collections import Counter, deque
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
from redis import Redis

from rtfm_agent import events as events_mod
from rtfm_agent import metrics as metrics_mod
from rtfm_agent.config import (
    CRAWL_DELAY_MS,
    CRAWL_HARD_PAGE_CAP,
    CRAWL_MAX_BYTES,
    CRAWL_MAX_DEPTH,
    CRAWL_MAX_PAGES,
    CRAWL_MIN_TEXT_CHARS,
    CRAWL_STAGE_TTL_H,
    CRAWL_TIMEOUT_S,
    CRAWL_ALLOW_PRIVATE_HOSTS,
    DOCS_DIR,
    ENABLE_WEB_CRAWL,
    WEB_DOCS_DIR,
)
from rtfm_agent.tenancy import TenantContext

logger = logging.getLogger(__name__)

USER_AGENT = "RTFMMeAgent/0.1 (+documentation indexer)"

_ALLOWED_SCHEMES = ("http", "https")
_ASSET_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".css", ".js", ".json", ".xml", ".rss", ".atom",
    ".zip", ".gz", ".tar", ".tgz", ".rar", ".7z",
    ".pdf", ".epub", ".mobi", ".doc", ".docx", ".ppt", ".pptx",
    ".mp3", ".mp4", ".wav", ".webm", ".mov", ".avi",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".exe", ".dmg", ".msi", ".deb", ".rpm", ".apk",
)
_TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid")
_HREF_RE = re.compile(r"""<a\s[^>]*?href=["']([^"'#]+)["']""", re.IGNORECASE)
_LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.IGNORECASE)

# One crawl at a time per tenant (in-process guard; matches the reingest
# threading model - single-process deployments only).
_ACTIVE: set[str] = set()
_ACTIVE_LOCK = threading.Lock()


class CrawlError(Exception):
    """Request-level rejection (bad URL, disabled feature, busy tenant)."""


class JobNotFoundError(CrawlError):
    """Unknown or already-expired job id."""


class JobStateError(CrawlError):
    """Operation invalid for the job's current lifecycle state."""


# --------------------------------------------------------------------------
# Paths / Redis keys
# --------------------------------------------------------------------------

def tenant_web_root(t: TenantContext) -> Path:
    """Approved crawled pages: docs/web/<org>/<host>/<slug>.md."""
    return Path(WEB_DOCS_DIR) / t.id


def staging_root(t: TenantContext) -> Path:
    """Staged-but-unreviewed jobs live OUTSIDE the approved tree so the
    ingestion loader (which globs docs/web/<org>/ recursively) can never
    pick up unapproved content."""
    return Path(WEB_DOCS_DIR) / "_staging" / t.id


def job_dir_for(t: TenantContext, job_id: str) -> Path:
    return staging_root(t) / job_id


def _record_key(t: TenantContext, job_id: str) -> str:
    return f"{t.prefix}crawl:{job_id}"


def _index_key(t: TenantContext) -> str:
    return f"{t.prefix}crawl:jobs"


# --------------------------------------------------------------------------
# URL handling / safety
# --------------------------------------------------------------------------

class SeedURL:
    """Validated seed: origin parts plus the optional same-prefix constraint."""

    def __init__(self, raw: str, path_prefix: str | None = None):
        parts = urlsplit(raw.strip())
        if parts.scheme not in _ALLOWED_SCHEMES:
            raise CrawlError("start_url must be an absolute http(s) URL")
        host = (parts.hostname or "").lower()
        if not host:
            raise CrawlError("start_url has no hostname")
        if parts.username or parts.password:
            raise CrawlError("credentials in start_url are not allowed")
        self.scheme = parts.scheme
        self.host = host
        self.netloc = parts.netloc
        self.url = urlunsplit((parts.scheme, parts.netloc, parts.path or "/", "", ""))

        prefix = (path_prefix or "").strip()
        if prefix and not prefix.startswith("/"):
            raise CrawlError("path_prefix must start with '/'")
        self.path_prefix = prefix.rstrip("/") or ""

    def allows(self, url: str) -> bool:
        """Same scheme + host (+ optional path prefix) policy."""
        try:
            parts = urlsplit(url)
        except ValueError:
            return False
        if parts.scheme != self.scheme or (parts.hostname or "").lower() != self.host:
            return False
        path = parts.path or "/"
        if self.path_prefix and not path.startswith(self.path_prefix):
            return False
        return True


def normalize_url(raw: str, base: str | None = None) -> str | None:
    """Absolute http(s) URL without fragment/tracking params, or None."""
    try:
        absolute = urljoin(base, raw.strip()) if base else raw.strip()
        parts = urlsplit(absolute)
    except ValueError:
        return None
    if parts.scheme not in _ALLOWED_SCHEMES or not parts.hostname:
        return None
    if parts.username or parts.password:
        return None
    query = "&".join(
        kv for kv in parts.query.split("&")
        if kv and not kv.split("=", 1)[0].lower().startswith(_TRACKING_PREFIXES)
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", query, ""))


def _is_asset(path: str) -> bool:
    return path.lower().endswith(_ASSET_SUFFIXES)


def _assert_public_host(host: str) -> None:
    """SSRF guard: refuse hosts resolving to non-public addresses."""
    if CRAWL_ALLOW_PRIVATE_HOSTS:
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise CrawlError(f"DNS resolution failed for '{host}': {exc}")
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if (
            addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_reserved or addr.is_multicast or addr.is_unspecified
        ):
            raise CrawlError(
                f"host '{host}' resolves to a private address ({addr}); "
                f"crawling it is blocked (CRAWL_ALLOW_PRIVATE_HOSTS overrides)"
            )


def page_id_for(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def _slug_for(url: str, pid: str) -> tuple[str, str]:
    """(host_dir, rel_path) for an approved page file."""
    parts = urlsplit(url)
    host_dir = re.sub(r"[^a-z0-9.-]", "_", (parts.hostname or "").lower())
    segments = [s for s in parts.path.split("/") if s]
    tail = segments[-1] if segments else "index"
    if tail.lower().endswith((".html", ".htm", ".php", ".asp", ".aspx")):
        tail = tail.rsplit(".", 1)[0]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", tail).strip("-") or "page"
    slug = slug[:60]
    return host_dir, f"{host_dir}/{slug}-{pid}.md"


# --------------------------------------------------------------------------
# Fetch helpers
# --------------------------------------------------------------------------

def _load_robots(client: httpx.Client, seed: SeedURL) -> RobotFileParser:
    parser = RobotFileParser()
    robots_url = f"{seed.scheme}://{seed.netloc}/robots.txt"
    try:
        resp = client.get(robots_url)
        if resp.status_code == 200:
            parser.parse(resp.text.splitlines())
        elif resp.status_code in (401, 403):
            # RFC 9309: unavailable due to server errors means assume disallow.
            parser.parse(["User-agent: *", "Disallow: /"])
        else:
            parser.parse([])  # 404 et al: crawling allowed
    except Exception as exc:
        logger.warning("robots.txt fetch failed for %s (allowing): %s",
                       robots_url, exc)
        parser.parse([])
    return parser


def _load_sitemap(client: httpx.Client, seed: SeedURL,
                  limit: int) -> list[str]:
    """<loc> URLs from /sitemap.xml (one level of sitemap-index expansion)."""
    found: list[str] = []

    def fetch_locs(url: str) -> tuple[list[str], str]:
        try:
            resp = client.get(url)
        except Exception:
            return [], ""
        if resp.status_code != 200:
            return [], ""
        body = resp.text
        lowered = body.lower()
        kind = ("sitemapindex" if "<sitemapindex" in lowered
                else "urlset" if "<urlset" in lowered else "urlset")
        locs = [_strip_html(m) for m in _LOC_RE.findall(body)]
        return locs, kind

    def collect(locs: list[str]) -> None:
        for loc in locs:
            normalized = normalize_url(loc)
            if normalized and seed.allows(normalized) and not _is_asset(
                    urlsplit(normalized).path):
                found.append(normalized)
                if len(found) >= limit:
                    return

    try:
        locs, kind = fetch_locs(f"{seed.scheme}://{seed.netloc}/sitemap.xml")
        if not locs:
            return found
        if kind == "sitemapindex":
            # sitemap index: descend into child sitemaps (bounded)
            for loc in locs:
                child = normalize_url(loc)
                if not child:
                    continue
                child_locs, _ = fetch_locs(child)
                collect(child_locs)
                if len(found) >= limit:
                    break
        else:
            collect(locs)
    except Exception as exc:
        logger.warning("sitemap discovery failed for %s: %s", seed.url, exc)
    return found


def _strip_html(value: str) -> str:
    return html_mod.unescape(re.sub(r"<!\[CDATA\[|\]\]>|<[^>]+>", "", value)).strip()


def extract_content(html: str, url: str) -> tuple[str, str]:
    """(title, markdown text) via trafilatura; empty text when unusable."""
    text = ""
    try:
        text = trafilatura.extract(
            html, url=url, output_format="markdown",
            include_links=False, include_tables=True,
        ) or ""
    except Exception as exc:
        logger.warning("trafilatura extraction failed for %s: %s", url, exc)
    title = ""
    try:
        meta = trafilatura.extract_metadata(html)
        title = (meta.title or "") if meta else ""
    except Exception:
        pass
    if not title:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        title = _strip_html(match.group(1)) if match else ""
    return title.strip()[:200], text.strip()


def _links_in(html: str) -> list[str]:
    return _HREF_RE.findall(html)


# --------------------------------------------------------------------------
# Job records
# --------------------------------------------------------------------------

def _write_record(r: Redis, t: TenantContext, job_id: str, **fields) -> None:
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


def _get_record(r: Redis, t: TenantContext, job_id: str) -> dict | None:
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
    """Drop staging dirs + records for jobs older than CRAWL_STAGE_TTL_H."""
    cutoff = time.time() - CRAWL_STAGE_TTL_H * 3600
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
    if not ENABLE_WEB_CRAWL:
        raise CrawlError("web crawling is disabled (ENABLE_WEB_CRAWL=0)")
    sweep_expired(r, t)

    seed = SeedURL(start_url, path_prefix=path_prefix)
    _assert_public_host(seed.host)

    with _ACTIVE_LOCK:
        if t.id in _ACTIVE:
            raise CrawlError(f"a crawl is already running for tenant '{t.id}'")
        _ACTIVE.add(t.id)

    job_id = uuid.uuid4().hex[:12]
    effective_pages = min(max_pages or CRAWL_MAX_PAGES, CRAWL_HARD_PAGE_CAP)
    effective_depth = CRAWL_MAX_DEPTH if max_depth is None else max_depth

    _write_record(r, t, job_id, status="running",
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
            timeout=CRAWL_TIMEOUT_S, follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        robots = _load_robots(client, seed)
        sitemap_budget = max(max_pages * 5, 100)
        sitemap_urls = _load_sitemap(client, seed, sitemap_budget)

        queue: deque[tuple[str, int]] = deque([(seed.url, 0)])
        seen = {seed.url}
        for url in sitemap_urls:
            if url not in seen:
                seen.add(url)
                queue.append((url, min(1, max_depth)))

        last_fetch_ts = 0.0
        delay_s = CRAWL_DELAY_MS / 1000.0

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

                body = resp.content[:CRAWL_MAX_BYTES]
                page_html = body.decode(
                    resp.encoding or "utf-8", errors="replace")
                title, text = extract_content(page_html, final_url)
                if len(text) < CRAWL_MIN_TEXT_CHARS:
                    skipped["thin_content"] += 1
                    continue

                pid = page_id_for(final_url)
                host_dir, rel_path = _slug_for(final_url, pid)
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
                    for href in _links_in(page_html):
                        link = normalize_url(href, base=final_url)
                        if (not link or link in seen or not seed.allows(link)
                                or _is_asset(urlsplit(link).path)):
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
        _write_record(r, t, job_id, status=status,
                      pages_total=str(len(seen)), pages_staged=str(len(pages)),
                      failures=str(failures),
                      summary_json=json.dumps(summary, ensure_ascii=False))
        metrics_mod.record_crawl(r, t, pages_fetched=fetched, failures=failures)
        events_mod.publish(
            r, t,
            events_mod.CRAWL_STAGED if pages else events_mod.CRAWL_FAILED,
            {"job_id": job_id, "tenant": t.id, **summary},
        )
    except Exception as exc:
        logger.exception("crawl job %s crashed", job_id)
        _write_record(r, t, job_id, status="failed", error=str(exc)[:500],
                      finished_at=str(time.time()))
        metrics_mod.record_crawl(r, t, pages_fetched=fetched, failures=failures + 1)
        events_mod.publish(r, t, events_mod.CRAWL_FAILED,
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
            # Lazy import: ingest pulls the embedder; keep crawler importable
            # without it. docs-dir precedence mirrors api._resolve_docs_dir
            # (docs/<org>/ first) without importing the api module (cycle).
            from rtfm_agent.ingest import run_ingestion

            approve_job(r, t, job_id, exclude_ids=None)
            team_dir = Path(__file__).resolve().parent.parent / "docs" / t.id
            docs_dir = str(team_dir) if team_dir.is_dir() else str(Path(DOCS_DIR))
            run_ingestion(r, t, docs_dir=docs_dir)
            mark_ingested(r, t, job_id)
        except Exception as exc:
            logger.error("auto-ingest after crawl %s failed: %s", job_id, exc)
            _write_record(r, t, job_id, status="failed",
                          error=str(exc)[:500], finished_at=str(time.time()))


# --------------------------------------------------------------------------
# Review operations (list / inspect / approve / discard)
# --------------------------------------------------------------------------

def list_jobs(r: Redis, t: TenantContext, limit: int = 20) -> list[dict]:
    try:
        ids = r.zrevrange(_index_key(t), 0, max(limit - 1, 0))
    except Exception as exc:
        logger.warning("crawl job listing failed: %s", exc)
        return []
    jobs = []
    for raw in ids:
        job_id = raw.decode() if isinstance(raw, bytes) else raw
        record = _get_record(r, t, job_id)
        if record:
            jobs.append({"job_id": job_id, **record})
    return jobs


def get_job(r: Redis, t: TenantContext, job_id: str) -> dict:
    record = _get_record(r, t, job_id)
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
    _write_record(r, t, job_id, status="approved", finished_at=str(time.time()),
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
    _write_record(r, t, job_id, status="ingested")


def discard_job(r: Redis, t: TenantContext, job_id: str) -> dict:
    """Throw a staged job away without touching anything approved."""
    record = _get_record(r, t, job_id)
    if record is None:
        raise JobNotFoundError(f"unknown crawl job '{job_id}'")
    if record.get("status") not in ("awaiting_review", "failed"):
        raise JobStateError(
            f"job '{job_id}' is '{record.get('status')}' and cannot be discarded")
    shutil.rmtree(job_dir_for(t, job_id), ignore_errors=True)
    _write_record(r, t, job_id, status="discarded", finished_at=str(time.time()))
    metrics_mod.record_crawl(r, t, discarded=True)
    return {"job_id": job_id, "status": "discarded"}
