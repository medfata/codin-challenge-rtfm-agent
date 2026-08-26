"""Step 13 verification: web crawl with staged review.

Part A: units - URL policy, normalization, asset filter, SSRF guard (no services).
Part B: live drills against a spawned uvicorn instance + a local fixture site
        served over real HTTP: crawl -> stage -> review endpoints -> approve
        with exclusion -> merged-corpus ingestion -> re-crawl removal ->
        discard, busy guard, tenant isolation, metrics.
Part C: key hygiene + cleanup.

Redis/HTTP drills degrade to [SKIP]; exit code 1 iff any FAIL.
"""

import os
import sys
import time

# Force the SSRF guard ON for Part A regardless of developer .env overrides;
# load_dotenv does not clobber pre-set env vars.
os.environ["CRAWL_ALLOW_PRIVATE_HOSTS"] = "0"

import shutil
import subprocess
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx
from redis import Redis

from rtfm_agent import crawler as crawler_mod
from rtfm_agent.config import settings
from rtfm_agent.common.tenancy import TenantContext

PASS = 0
FAIL = 0
TENANT_A = "step13a"
TENANT_B = "step13b"
PORT = 8012
BASE = f"http://localhost:{PORT}"
HDRS_A = {"X-Tenant-Id": TENANT_A}
MARKER_INTRO = "RTFMINTRO-MARKER-42"
SLUG_ADVANCED_SUBSTR = "advanced"


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"[PASS] {name}" + (f" - {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"[FAIL] {name}" + (f" - {detail}" if detail else ""))


def skip(name: str, detail: str = "") -> None:
    print(f"[SKIP] {name}" + (f" - {detail}" if detail else ""))


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html><head><title>"
        f"{title}</title></head><body><h1>{title}</h1>\n{body}\n</body></html>"
    )


def _rich(marker: str, extra: str = "") -> str:
    paragraphs = "\n".join(
        f"<p>Paragraph {i}: {marker} documents how retrieval augmented "
        f"generation pipelines chunk source material, embed the pieces, and "
        f"answer questions strictly from indexed context.</p>"
        for i in range(1, 6)
    )
    return _page(marker, f"<p>{marker} overview section.</p>\n{paragraphs}{extra}")


def build_fixture(root: Path) -> None:
    """Mini doc site: linked pages, sitemap, robots-disallowed path, traps."""
    guide = root / "guide"
    guide.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(_page(
        "Fixture Docs Home",
        f"<p>Welcome to {MARKER_INTRO} documentation home.</p>"
        '<p>Start with <a href="guide/intro.html">the intro</a> or '
        '<a href="guide/advanced.html">advanced usage</a>. '
        "This index explains crawling, staging, and review in depth for "
        "verification purposes and repeats itself to clear the length bar. "
        'Off-site trap: <a href="http://example.invalid/never">never follow</a>.</p>',
    ), encoding="utf-8")
    (guide / "intro.html").write_text(_rich(MARKER_INTRO), encoding="utf-8")
    (guide / "advanced.html").write_text(
        _rich("ADVANCED-MARKER-7"), encoding="utf-8")
    (guide / "blocked.html").write_text(
        _rich("BLOCKED-MARKER-9"), encoding="utf-8")
    (root / "thin.html").write_text(
        _page("Thin page", "<p>Too short.</p>"), encoding="utf-8")
    urls = "\n".join(
        f"<loc>http://localhost:{PORT}/{p}</loc>"
        for p in ("index.html", "guide/intro.html", "guide/advanced.html",
                  "guide/blocked.html", "thin.html")
    )
    (root / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
        f"{urls}</urlset>", encoding="utf-8")
    (root / "robots.txt").write_text(
        "User-agent: *\nDisallow: /guide/blocked\n", encoding="utf-8")


def serve_fixture(root: Path):
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, port


def rewrite_fixture_ports(root: Path, old_port: int, new_port: int) -> None:
    for path in list(root.rglob("*.html")) + [root / "sitemap.xml"]:
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(f":{old_port}", f":{new_port}"),
                        encoding="utf-8")


def _spawn_server(port: int, extra_env: dict) -> tuple[subprocess.Popen, Path]:
    log_path = Path(tempfile.mkdtemp(prefix="rtfm-step13-srv-")) / "uvicorn.log"
    env = {**os.environ, **extra_env}
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "rtfm_agent.api:app",
             "--port", str(port), "--log-level", "warning"],
            cwd=str(project_root), env=env,
            stdout=log_file, stderr=log_file,
        )
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if httpx.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                return proc, log_path
        except Exception:
            time.sleep(0.7)
    proc.terminate()
    raise RuntimeError(f"uvicorn on :{port} did not become healthy")


def wait_job(client: httpx.Client, job_id: str, want: str,
             timeout_s: float = 45.0) -> dict:
    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        resp = client.get(f"{BASE}/crawl/jobs/{job_id}", headers=HDRS_A)
        last = _json(resp)
        if last.get("status") == want:
            return last
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} never reached '{want}': {last}")


def _json(resp: httpx.Response) -> dict:
    """Parse JSON but surface raw payload on non-JSON errors."""
    try:
        return resp.json()
    except ValueError:
        print(f"[INFO] non-JSON response {resp.status_code}: "
              f"{resp.text[:300]!r}")
        return {"_status": resp.status_code}


def part_a_units() -> None:
    seed = crawler_mod.SeedURL("https://docs.example.com/guide/start")
    check("SeedURL accepts absolute https", seed.url.endswith("/guide/start"))
    for bad in ("ftp://example.com/x", "not a url", "http://user:pw@example.com/"):
        try:
            crawler_mod.SeedURL(bad)
            check(f"SeedURL rejects {bad!r}", False, "no error")
        except crawler_mod.CrawlError:
            check(f"SeedURL rejects {bad!r}", True)
    try:
        crawler_mod.SeedURL("https://example.com/a", path_prefix="docs")
        check("path_prefix must start with '/'", False, "no error")
    except crawler_mod.CrawlError:
        check("path_prefix must start with '/'", True)

    check("same-host policy admits sibling paths",
          seed.allows("https://docs.example.com/guide/deep/page.html"))
    check("same-host policy rejects other hosts",
          not seed.allows("https://other.example.com/guide/x"))
    constrained = crawler_mod.SeedURL("https://docs.example.com/guide/start",
                                      path_prefix="/guide")
    check("path-prefix policy enforced when requested",
          not constrained.allows("https://docs.example.com/blog/x")
          and constrained.allows("https://docs.example.com/guide/deep"))

    norm = crawler_mod.normalize_url("/a/b?utm_source=x&id=3#frag",
                                     base="https://e.com/")
    check("normalize_url strips fragments + tracking params",
          norm == "https://e.com/a/b?id=3", f"{norm}")
    check("asset filter catches binaries + scripts",
          crawler_mod._is_asset("/x.PNG") and crawler_mod._is_asset("/s/app.js")
          and not crawler_mod._is_asset("/guide/page.html"))

    try:
        crawler_mod._assert_public_host("localhost")
        check("SSRF guard blocks loopback by default", False, "no error")
    except crawler_mod.CrawlError:
        check("SSRF guard blocks loopback by default", True)

    saved = crawler_mod.CRAWL_ALLOW_PRIVATE_HOSTS
    try:
        crawler_mod.CRAWL_ALLOW_PRIVATE_HOSTS = True
        crawler_mod._assert_public_host("localhost")
        check("CRAWL_ALLOW_PRIVATE_HOSTS lifts the guard (tests only)", True)
    finally:
        crawler_mod.CRAWL_ALLOW_PRIVATE_HOSTS = saved


def part_b_live(r: Redis, fixture: Path, fixture_port: int) -> None:
    t_ctx = TenantContext(TENANT_A)
    seed_url = f"http://localhost:{fixture_port}/index.html"

    # Tiny team corpus so each approval's re-ingestion is fast + deterministic
    # (docs/<tenant>/ takes precedence over the 91-file progit default).
    team_dir = project_root / "docs" / TENANT_A
    team_dir.mkdir(parents=True, exist_ok=True)
    (team_dir / "local-alpha.asc").write_text(
        "== Alpha\n\nAlpha covers local vector search basics.\n\n"
        "A second paragraph keeps the chunker happy.\n", encoding="utf-8")
    (team_dir / "local-beta.asc").write_text(
        "== Beta\n\nBeta covers semantic caching behaviour.\n\n"
        "More text so extraction and chunking have room to work.\n",
        encoding="utf-8")

    # Busy guard at module level before any HTTP traffic.
    crawler_mod._ACTIVE.add(TENANT_A)
    try:
        crawler_mod.start_job(r, t_ctx, "https://localhost:1/x")
        check("one crawl per tenant is enforced", False, "no error")
    except crawler_mod.CrawlError:
        check("one crawl per tenant is enforced", True)
    finally:
        crawler_mod._ACTIVE.discard(TENANT_A)

    try:
        proc, server_log = _spawn_server(PORT, {
            "CRAWL_ALLOW_PRIVATE_HOSTS": "1",  # localhost fixture only
            "CRAWL_DELAY_MS": "25",
        })
    except Exception as exc:
        skip("live crawl drills", f"could not start uvicorn: {exc}")
        return

    client = httpx.Client(timeout=300)
    try:
        resp = client.get(f"{BASE}/crawl/review")
        check("review UI served same-origin",
              resp.status_code == 200
              and "text/html" in resp.headers.get("content-type", "")
              and "Crawl Review" in resp.text)

        resp = client.post(f"{BASE}/crawl", headers=HDRS_A,
                           json={"start_url": seed_url})
        check("POST /crawl accepts and stages async", resp.status_code == 202,
              resp.text[:100])
        job_id = resp.json()["job_id"]

        job = wait_job(client, job_id, "awaiting_review")
        pages = {p["url"]: p["page_id"] for p in job.get("pages", [])}
        staged_urls = set(pages)
        check("staged exactly the usable pages",
              len(pages) == 3
              and all(f"/index" in u or "/intro" in u or "/advanced" in u
                      for u in staged_urls),
              f"staged={sorted(staged_urls)}")

        summary = job.get("summary") or {}
        skipped = summary.get("skipped") or {}
        check("robots-disallowed page never fetched",
              skipped.get("robots_blocked", 0) >= 1, f"skipped={skipped}")
        check("thin page dropped on extraction quality",
              skipped.get("thin_content", 0) >= 1, f"skipped={skipped}")
        check("off-host link trap ignored",
              all("example.invalid" not in u for u in staged_urls))

        status = client.get(f"{BASE}/docs/status", headers=HDRS_A).json()
        check("corpus untouched while awaiting review",
              status.get("corpus") is None,
              f"corpus={status.get('corpus')}")

        intro_pid = next(pid for u, pid in pages.items() if u.endswith("intro.html"))
        preview = client.get(f"{BASE}/crawl/jobs/{job_id}/pages/{intro_pid}",
                             headers=HDRS_A).json()
        check("preview returns extracted text for verification",
              MARKER_INTRO in preview.get("text", ""), f"chars={preview.get('chars')}")

        advanced_pid = next((pid for u, pid in pages.items()
                             if u.endswith("advanced.html")), None)
        # Approve EVERYTHING first so all three pages join the corpus.
        resp = client.post(f"{BASE}/crawl/jobs/{job_id}/approve",
                           headers=HDRS_A, json={"exclude": []})
        approved = _json(resp)
        ing = approved.get("ingestion") or {}
        check("approval ingests all approved pages",
              resp.status_code == 200 and approved.get("approved") == 3,
              f"approved={approved.get('approved')}")
        check("approval triggers full ingestion",
              ing.get("chunks_stored", 0) > 0 and ing.get("corpus_version", 0) >= 1,
              f"v={ing.get('corpus_version')} chunks={ing.get('chunks_stored')}")

        from rtfm_agent import llm as llm_client
        if not settings.llm.api_key:
            skip("ask cites web sources", "LLM key not configured")
        else:
            resp = client.post(f"{BASE}/ask", headers=HDRS_A,
                               json={"question":
                                     f"What does {MARKER_INTRO} document?"})
            citations = resp.json().get("citations", [])
            sources = [c.get("source_file", "") for c in citations]
            check("answers cite crawled pages among merged-corpus sources",
                  any(s.startswith("web/") for s in sources),
                  f"sources={sources[:3]}")

        def _web_keys():
            return [k.decode()
                    for k in r.scan_iter(match=f"t:{TENANT_A}:doc:web/*")]

        web_keys = _web_keys()
        has_advanced = any(SLUG_ADVANCED_SUBSTR in k for k in web_keys)
        check("approved pages indexed under web/ namespace",
              len(web_keys) > 0 and has_advanced,
              f"web_chunks={len(web_keys)}")

        # Re-crawl: exclude intro via review, and advanced vanishes from the
        # source site - approval must remove BOTH from the corpus.
        (fixture / "guide" / "advanced.html").unlink()
        resp = client.post(f"{BASE}/crawl", headers=HDRS_A,
                           json={"start_url": seed_url})
        job2 = resp.json()["job_id"]
        job2_data = wait_job(client, job2, "awaiting_review")
        pages2 = {p["url"]: p["page_id"] for p in job2_data.get("pages", [])}
        check("vanished page no longer staged",
              len(pages2) == 2
              and not any(u.endswith("advanced.html") for u in pages2),
              f"staged={sorted(pages2)}")
        intro_pid = next((pid for u, pid in pages2.items()
                          if u.endswith("intro.html")), None)
        resp = client.post(f"{BASE}/crawl/jobs/{job2}/approve",
                           headers=HDRS_A, json={"exclude": [intro_pid]})
        approved2 = _json(resp)
        removed = approved2.get("removed_files", 0)
        version2 = (approved2.get("ingestion") or {}).get("corpus_version", 0)
        check("re-crawl removes vanished + excluded pages from the corpus",
              removed >= 2 and version2 >= 2,
              f"removed={removed} v={version2}")

        remaining = _web_keys()
        gone = [k for k in remaining
                if SLUG_ADVANCED_SUBSTR in k or "/intro-" in k]
        check("only still-approved pages remain indexed",
              bool(remaining) and not gone, f"remaining={len(remaining)}")

        # Discard path leaves zero trace.
        resp = client.post(f"{BASE}/crawl", headers=HDRS_A,
                           json={"start_url": f"http://localhost:{fixture_port}/thin.html",
                                 "max_pages": 2})
        job3 = resp.json()["job_id"]
        wait_job(client, job3, "awaiting_review", timeout_s=20)
        staging_dir = (Path(os.getenv("WEB_DOCS_DIR", "docs/web"))
                       / "_staging" / TENANT_A / job3)
        resp = client.delete(f"{BASE}/crawl/jobs/{job3}", headers=HDRS_A)
        check("discard deletes staged payload",
              resp.status_code == 200
              and resp.json().get("status") == "discarded"
              and not staging_dir.exists())

        resp = client.get(f"{BASE}/crawl/jobs/{job_id}", headers=HDRS_A).json()
        check("approved job record retained as audit",
              resp.get("status") == "ingested")

        # Tenant isolation.
        resp = client.get(f"{BASE}/crawl/jobs", headers={"X-Tenant-Id": TENANT_B})
        check("other tenant sees no crawl jobs", resp.json().get("jobs") == [])
        status_b = client.get(f"{BASE}/docs/status",
                              headers={"X-Tenant-Id": TENANT_B}).json()
        check("other tenant has no corpus", status_b.get("corpus") is None)

        snap = client.get(f"{BASE}/metrics", headers=HDRS_A).json()
        check("crawl counters recorded",
              snap.get("crawl_jobs_total", 0) >= 3
              and snap.get("crawl_pages_fetched_total", 0) > 0,
              f"jobs={snap.get('crawl_jobs_total')} "
              f"fetched={snap.get('crawl_pages_fetched_total')}")
    finally:
        client.close()
        if proc:
            proc.terminate()
            time.sleep(1.0)
        if FAIL and server_log.is_file():
            print(f"[INFO] uvicorn log tail ({server_log}):")
            for line in server_log.read_text(encoding="utf-8",
                                             errors="replace").splitlines()[-40:]:
                print(f"       {line}")


def part_c_cleanup(r: Redis, fixture: Path) -> None:
    cleaned = 0
    try:
        for tenant in (TENANT_A, TENANT_B):
            keys = list(r.scan_iter(match=f"t:{tenant}:*"))
            if keys:
                r.delete(*keys)
                cleaned += len(keys)
    except Exception as exc:
        skip("key hygiene cleanup", str(exc))
    stray = []
    try:
        for match in ("crawl*", "*_staging*"):
            stray += [k.decode() for k in r.scan_iter(match=match)
                      if not k.decode().startswith(("t:",))]
    except Exception:
        pass
    check("all crawl keys tenant-scoped", not stray, f"stray={stray[:5]}")
    for path in (project_root / "docs" / "web" / TENANT_A,
                 project_root / "docs" / "web" / TENANT_B,
                 project_root / "docs" / "web" / "_staging" / TENANT_A,
                 project_root / "docs" / "web" / "_staging" / TENANT_B,
                 project_root / "docs" / TENANT_A,
                 project_root / "docs" / TENANT_B):
        shutil.rmtree(path, ignore_errors=True)
    shutil.rmtree(fixture, ignore_errors=True)
    print(f"[INFO] cleaned {cleaned} redis keys + docs/web scratch dirs")


def main() -> int:
    print("=" * 64)
    print("Step 13 verification: web crawl with staged review")
    print("=" * 64)

    part_a_units()

    fixture = Path(tempfile.mkdtemp(prefix="rtfm-crawl-fixture-"))
    build_fixture(fixture)
    server, fixture_port = serve_fixture(fixture)
    # The sitemap hardcodes its own absolute URLs; point them at the live port.
    rewrite_fixture_ports(fixture, PORT, fixture_port)

    r = None
    try:
        r = Redis.from_url(settings.redis.url, decode_responses=False)
        r.ping()
    except Exception as exc:
        skip("live drills", f"Redis unreachable: {exc}")

    try:
        if r is not None:
            part_b_live(r, fixture, fixture_port)
    finally:
        server.shutdown()
        server.server_close()
        if r is not None:
            part_c_cleanup(r, fixture)
            r.close()

    print("-" * 64)
    print(f"PASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

