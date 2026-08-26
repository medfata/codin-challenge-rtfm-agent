"""Crawl fetching layer: seed validation, SSRF guard, robots.txt, sitemap
discovery, link extraction, and readable-text extraction.

Safety rails:
  * http/https only, credentials-in-URL rejected
  * SSRF guard: DNS-resolve the host and reject private/loopback/link-local/
    reserved targets (crawl.allow_private_hosts lifts this for tests ONLY)
"""

import hashlib
import html as html_mod
import ipaddress
import logging
import re
import socket
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from rtfm_agent.config import settings

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


class CrawlError(Exception):
    """Request-level rejection (bad URL, disabled feature, busy tenant)."""


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


def is_asset(path: str) -> bool:
    """True when the URL path points at a non-document asset."""
    return path.lower().endswith(_ASSET_SUFFIXES)


def assert_public_host(host: str) -> None:
    """SSRF guard: refuse hosts resolving to non-public addresses."""
    if settings.crawl.allow_private_hosts:
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


def slug_for(url: str, pid: str) -> tuple[str, str]:
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


def strip_html(value: str) -> str:
    return html_mod.unescape(re.sub(r"<!\[CDATA\[|\]\]>|<[^>]+>", "", value)).strip()


def links_in(html: str) -> list[str]:
    return _HREF_RE.findall(html)


# --------------------------------------------------------------------------
# Fetch helpers
# --------------------------------------------------------------------------


def load_robots(client: httpx.Client, seed: SeedURL) -> RobotFileParser:
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


def load_sitemap(client: httpx.Client, seed: SeedURL, limit: int) -> list[str]:
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
        locs = [strip_html(m) for m in _LOC_RE.findall(body)]
        return locs, kind

    def collect(locs: list[str]) -> None:
        for loc in locs:
            normalized = normalize_url(loc)
            if normalized and seed.allows(normalized) and not is_asset(
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
        title = strip_html(match.group(1)) if match else ""
    return title.strip()[:200], text.strip()
