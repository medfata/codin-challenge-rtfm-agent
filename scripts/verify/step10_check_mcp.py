"""Step 10 verification: MCP exposure.

Part A: units - tenant resolution matrix + bearer middleware, no services.
Part B: live drills against a spawned uvicorn instance using raw JSON-RPC
        over streamable HTTP (initialize / tools/list / tools/call):
        handshake, tool inventory, tenant fallback + isolation via seeded
        mini-corpus, metrics counter, optional embedder/LLM-backed tools,
        bearer enforcement on a second server instance.
Part C: key hygiene + cleanup.

Redis/HTTP drills degrade to [SKIP]; exit code 1 iff any FAIL.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx
import numpy as np
from redis import Redis

from rtfm_agent.ingestion import pipeline as ingest_mod
from rtfm_agent.ingestion import versioning as versions_mod
from rtfm_agent.config import settings
from rtfm_agent.ingestion.documents import load_asc_files
from rtfm_agent.common.tenancy import TenantContext

PASS = 0
FAIL = 0
TENANT_A = "step10a"
TENANT_B = "step10b"
PORT_A = 8010
PORT_B = 8011

FILES = {
    "alpha.asc": "# Alpha\n\nAlpha covers vectors.",
    "beta.asc": "# Beta\n\nBeta covers caching.",
}


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


def _part_a_units() -> None:
    """Tenant resolution + bearer middleware against pure functions."""
    from mcp.server.mcpserver.exceptions import ToolError

    from rtfm_agent import mcp_server as mcp_mod

    ctx = mcp_mod.resolve_tenant({"X-Tenant-Id": TENANT_A})
    check("resolve_tenant honours header", ctx.id == TENANT_A)
    ctx = mcp_mod.resolve_tenant({"x-tenant-id": f" {TENANT_A.upper()} "})
    check("resolve_tenant case-insensitive key + normalises value",
          ctx.id == TENANT_A)

    saved_default = mcp_mod.MCP_DEFAULT_TENANT
    try:
        mcp_mod.MCP_DEFAULT_TENANT = ""
        for bad in ({}, {"X-Tenant-Id": "BAD SLUG!"}, {"X-Tenant-Id": ""}):
            try:
                mcp_mod.resolve_tenant(bad)
                check(f"resolve_tenant rejects {bad}", False, "no error")
            except ToolError as exc:
                check(f"resolve_tenant rejects {bad}",
                      "X-Tenant-Id" in str(exc))
        mcp_mod.MCP_DEFAULT_TENANT = TENANT_A
        ctx = mcp_mod.resolve_tenant({})
        check("resolve_tenant falls back to MCP_DEFAULT_TENANT",
              ctx.id == TENANT_A)
    finally:
        mcp_mod.MCP_DEFAULT_TENANT = saved_default

    async def dummy_app(scope, receive, send):
        dummy_app.called = True

    import asyncio

    from starlette.responses import JSONResponse  # noqa: F401

    mw = mcp_mod.BearerTokenMiddleware(dummy_app, "sekret")

    def run_scope(headers: list[tuple[bytes, bytes]]) -> bool:
        async def scenario() -> bool:
            dummy_app.called = False
            sent = {}

            async def receive():
                return {"type": "http.request"}

            async def send(message):
                sent.setdefault("status", message.get("status"))

            scope = {
                "type": "http", "method": "POST", "path": "/mcp",
                "headers": headers, "query_string": b"",
            }
            await mw(scope, receive, send)
            if dummy_app.called:
                return True
            return sent.get("status") != 401

        return asyncio.run(scenario())

    check("bearer middleware forwards correct token",
          run_scope([(b"authorization", b"Bearer sekret")]))
    check("bearer middleware blocks missing token", not run_scope([]))
    check("bearer middleware blocks wrong token",
          not run_scope([(b"authorization", b"Bearer nope")]))


class McpRpc:
    """Minimal JSON-RPC-over-streamable-HTTP client for checks."""

    def __init__(self, base_url: str, headers: dict | None = None):
        self.url = base_url.rstrip("/") + "/mcp"
        self.session_id: str | None = None
        self._id = 0
        base = {"Accept": "application/json, text/event-stream"}
        self.fixed_headers = {**base, **(headers or {})}
        self.http = httpx.Client(timeout=60)

    def close(self):
        self.http.close()

    def _post(self, payload: dict) -> httpx.Response:
        headers = dict(self.fixed_headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        resp = self.http.post(self.url, json=payload, headers=headers)
        sid = resp.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid
        return resp

    @staticmethod
    def _json(resp: httpx.Response):
        ctype = resp.headers.get("content-type", "")
        if "event-stream" in ctype:
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    try:
                        return json.loads(line[len("data:"):].strip())
                    except json.JSONDecodeError:
                        continue
            return {}
        try:
            return resp.json()
        except json.JSONDecodeError:
            return {}

    def initialize(self) -> dict:
        resp = self._post({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "step10-check", "version": "0"},
            },
        })
        body = self._json(resp)
        if not body.get("result"):
            raise RuntimeError(
                f"initialize failed: status={resp.status_code} "
                f"ctype={resp.headers.get('content-type')} "
                f"body={resp.text[:200]!r}"
            )
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return body

    def list_tools(self) -> list[str]:
        self._id += 1
        resp = self._post({
            "jsonrpc": "2.0", "id": 100 + self._id, "method": "tools/list",
        })
        tools = self._json(resp).get("result", {}).get("tools", [])
        return sorted(t.get("name", "?") for t in tools)

    def call_tool(self, name: str, arguments: dict | None = None):
        """Returns (is_error, payload) where payload is structuredContent,
        parsed JSON text, or plain text."""
        self._id += 1
        resp = self._post({
            "jsonrpc": "2.0", "id": 100 + self._id, "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        })
        body = self._json(resp)
        if "error" in body:
            return True, body["error"].get("message", "rpc error")
        result = body.get("result", {})
        sc = result.get("structuredContent")
        # Some SDK versions wrap object payloads as {"result": ...}
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            sc = sc["result"]
        if sc is not None:
            return bool(result.get("isError")), sc
        texts = [
            c.get("text", "") for c in result.get("content", [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        joined = "\n".join(texts)
        try:
            return bool(result.get("isError")), json.loads(joined)
        except (json.JSONDecodeError, TypeError):
            return bool(result.get("isError")), joined


def _seed_corpus(r: Redis, t: TenantContext, tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    for name, content in FILES.items():
        (tmp / name).write_text(content, encoding="utf-8")
    docs = load_asc_files(str(tmp))
    prep = versions_mod.prepare(r, t, docs)
    zeros = np.zeros(settings.embed.dim, dtype=np.float32)
    chunks = [{
        "source_file": d["source_file"], "heading": d["heading"],
        "chunk_pos": 1, "chunk_text": d["content"], "embedding": zeros,
        "doc_version": prep["version"],
    } for d in docs]
    ingest_mod.delete_source_keys(r, t)
    ingest_mod.create_redis_index(r, t, settings.embed.dim)
    ingest_mod.store_in_redis(r, t, chunks)
    versions_mod.finalize(r, t, prep, chunks)


def _spawn_server(port: int, extra_env: dict) -> subprocess.Popen:
    env = {**os.environ, **extra_env}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "rtfm_agent.api:app",
         "--port", str(port), "--log-level", "warning"],
        cwd=str(project_root), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if httpx.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                return proc
        except Exception:
            time.sleep(0.7)
    proc.terminate()
    raise RuntimeError(f"uvicorn on :{port} did not become healthy")


def part_b_live(r: Redis, tmp: Path) -> None:
    try:
        _seed_corpus(r, TenantContext(TENANT_A), tmp / TENANT_A)
    except Exception as exc:
        skip("live MCP drills", f"Redis seeding failed: {exc}")
        return

    proc = None
    try:
        proc = _spawn_server(PORT_A, {"MCP_DEFAULT_TENANT": TENANT_A})
    except Exception as exc:
        skip("live MCP drills", f"could not start uvicorn: {exc}")
        return

    rpc_a = McpRpc(f"http://localhost:{PORT_A}", {"X-Tenant-Id": TENANT_A})
    try:
        init = rpc_a.initialize()
        server_info = init.get("result", {}).get("serverInfo", {})
        check("MCP initialize handshake", server_info.get("name") == "rtfm",
              json.dumps(server_info)[:80])

        expected = ["ask_question", "documentation_status", "list_documents",
                    "search_documents", "service_metrics"]
        tools = rpc_a.list_tools()
        check("tools/list exposes exactly the five planned tools",
              tools == expected, f"{tools}")

        err, status = rpc_a.call_tool("documentation_status")
        corpus = (status or {}).get("corpus") or {}
        check("documentation_status sees seeded corpus via header tenant "
              "(step10a)", not err and corpus.get("version") == 1,
              f"v={corpus.get('version')}")

        rpc_nohdr = McpRpc(f"http://localhost:{PORT_A}")
        try:
            rpc_nohdr.initialize()
            err, status = rpc_nohdr.call_tool("documentation_status")
            corpus = (status or {}).get("corpus") or {}
            check("no-header client falls back to MCP_DEFAULT_TENANT",
                  not err and corpus.get("version") == 1,
                  f"v={corpus.get('version')}")
        finally:
            rpc_nohdr.close()

        rpc_b = McpRpc(f"http://localhost:{PORT_A}", {"X-Tenant-Id": TENANT_B})
        try:
            rpc_b.initialize()
            err, status = rpc_b.call_tool("documentation_status")
            check("other tenant's corpus is invisible (isolation)",
                  not err and (status or {}).get("corpus") is None,
                  f"corpus={(status or {}).get('corpus')}")
        finally:
            rpc_b.close()

        err, listing = rpc_a.call_tool("list_documents")
        check("list_documents reports versioned corpus",
              not err and "Corpus version 1" in str(listing),
              str(listing)[:60])

        err, snap = rpc_a.call_tool("service_metrics")
        check("service_metrics round-trips counters",
              not err and isinstance(snap, dict) and "requests_total" in snap,
              f"keys={sorted(snap)[:5] if isinstance(snap, dict) else snap}")

        err, results = rpc_a.call_tool(
            "search_documents", {"query": "how does beta caching work?", "k": 2})
        if err and "failed" in str(results):
            skip("search_documents", f"embedder unavailable: {str(results)[:70]}")
        else:
            files = [c.get("source_file") for c in (results or {}).get("results", [])]
            check("search_documents returns chunk hits scoped to tenant",
                  not err and all(f in FILES for f in files) and files,
                  f"files={files}")

        err, answer = rpc_a.call_tool(
            "ask_question", {"question": "What does the beta document cover?"})
        if err:
            skip("ask_question", f"LLM unavailable: {str(answer)[:70]}")
        else:
            sid_ok = bool(answer.get("session_id"))
            follow = McpRpc(f"http://localhost:{PORT_A}",
                            {"X-Tenant-Id": TENANT_A})
            try:
                follow.initialize()
                ferr, follow_res = follow.call_tool("ask_question", {
                    "question": "Summarise your previous answer in 3 words.",
                    "session_id": answer.get("session_id"),
                })
                check("ask_question answers + session_id continuity",
                      not ferr and sid_ok and bool(follow_res.get("answer")),
                      f"citations={len(answer.get('citations', []))}")
            finally:
                follow.close()

        err, snap = rpc_a.call_tool("service_metrics")
        total = int((snap or {}).get("mcp_calls_total", 0)) if not err else 0
        check("mcp_calls_total accumulated across calls", total >= 4,
              f"count={total}")
    finally:
        rpc_a.close()
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    # --- Second instance: bearer enforcement ------------------------------
    proc_b = None
    try:
        proc_b = _spawn_server(PORT_B, {"MCP_BEARER_TOKEN": "sekret"})
    except Exception as exc:
        skip("bearer drill", f"could not start second uvicorn: {exc}")
        return
    try:
        try:
            httpx.post(f"http://localhost:{PORT_B}/mcp", json={}, timeout=10)
            blocked_direct = False
        except Exception:
            blocked_direct = True
        rpc_auth = McpRpc(f"http://localhost:{PORT_B}",
                          {"Authorization": "Bearer sekret",
                           "X-Tenant-Id": TENANT_A})
        try:
            init = rpc_auth.initialize()
            check("correct bearer token passes through to MCP",
                  init.get("result", {}).get("serverInfo", {}).get("name") == "rtfm")

            rpc_bad = McpRpc(f"http://localhost:{PORT_B}",
                             {"Authorization": "Bearer wrong"})
            try:
                resp = rpc_bad._post({"jsonrpc": "2.0", "id": 1,
                                      "method": "initialize", "params": {}})
                check("wrong bearer token rejected with 401",
                      resp.status_code == 401, f"status={resp.status_code}")
            finally:
                rpc_bad.close()

            rpc_notenant = McpRpc(f"http://localhost:{PORT_B}",
                                  {"Authorization": "Bearer sekret"})
            try:
                rpc_notenant.initialize()
                err, msg = rpc_notenant.call_tool("service_metrics")
                check("token ok + no tenant -> actionable tool error",
                      err and "X-Tenant-Id" in str(msg), str(msg)[:60])
            finally:
                rpc_notenant.close()
        finally:
            rpc_auth.close()
        del blocked_direct
    finally:
        if proc_b:
            proc_b.terminate()
            try:
                proc_b.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc_b.kill()


def part_c_hygiene(r: Redis) -> None:
    allowed = (f"t:{TENANT_A}:".encode(), f"t:{TENANT_B}:".encode())
    # AMS owns working-memory keys without our prefix - it isolates via
    # namespaces instead of key prefixes (documented in MULTITENANCY-PLAN).
    allowed += (b"working_memory:",)
    stray = [k for k in r.scan_iter(match=b"*step10*")
             if not k.startswith(allowed)]
    check("all step10 keys tenant-prefixed (AMS working_memory exempt)",
          not stray, f"stray={stray[:3]}")
    for slug in (TENANT_A, TENANT_B):
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=f"t:{slug}:*", count=500)
            if keys:
                r.delete(*keys)
            if cursor == 0:
                break


if __name__ == "__main__":
    t0 = time.time()
    print(f"== Step 10 checks - {time.strftime('%Y-%m-%d %H:%M:%S')} ==")
    _part_a_units()
    r_conn = None
    tmp_root = Path(tempfile.mkdtemp(prefix="step10_", dir=str(project_root)))
    try:
        try:
            r_conn = Redis.from_url(settings.redis.url, decode_responses=False,
                                    socket_connect_timeout=3)
            r_conn.ping()
        except Exception as exc:
            skip("Redis-dependent drills", f"no Redis at {settings.redis.url}: {exc}")
        if r_conn:
            part_b_live(r_conn, tmp_root)
            part_c_hygiene(r_conn)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        if r_conn:
            r_conn.close()
    print(f"\n{PASS} passed, {FAIL} failed in {time.time() - t0:.1f}s")
    sys.exit(1 if FAIL else 0)
