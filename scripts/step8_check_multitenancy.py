"""Step 8 verification: multi-tenancy isolation.

Part A: tenancy unit checks (validation + FastAPI dependency), no services.
Part B: Redis isolation drills across two synthetic tenants (docs, cache,
        sessions, metrics, scoped flush, key hygiene). No LLM, no embedder -
        query vectors reuse the stored float32 blob so cosine distance ~ 0.
Part C: live HTTP probes against RTFM_API_URL (skipped when unreachable).

Redis drills and API probes degrade to [SKIP]; exit code 1 iff any FAIL.
"""

import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx
import numpy as np
from fastapi import HTTPException
from redis import Redis

from rtfm_agent import cache as cache_mod
from rtfm_agent import ingest as ingest_mod
from rtfm_agent import metrics as metrics_mod
from rtfm_agent import retrieval
from rtfm_agent import sessions as sessions_mod
from rtfm_agent.config import EMBEDDING_DIM, REDIS_URL, TENANTS_OPEN, TENANT_ALLOWLIST
from rtfm_agent.tenancy import TenantContext, normalize_tenant, require_tenant

PASS = 0
FAIL = 0

TENANT_A = "step8a"
TENANT_B = "step8b"
API_BASE = os.getenv("RTFM_API_URL", "http://localhost:8000").rstrip("/")
ALLOWED_PREFIXES = (f"t:{TENANT_A}:".encode(), f"t:{TENANT_B}:".encode())
LEGACY_PREFIXES = (b"doc:", b"cache:", b"session:", b"metrics:")


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


def check_tenancy_units() -> None:
    """Part A: pure validation/derivation checks, no external services."""
    acme = normalize_tenant("Acme")
    if TENANTS_OPEN or "acme" in TENANT_ALLOWLIST:
        check("normalize_tenant('Acme') lowercases to id 'acme'",
              acme is not None and acme.id == "acme", f"ctx={acme!r}")
        check("TenantContext derives prefix/doc/cache/metrics names",
              acme is not None
              and acme.prefix == "t:acme:"
              and acme.doc_index == "t:acme:doc_idx"
              and acme.cache_index == "t:acme:cache_idx"
              and acme.metrics_key == "t:acme:metrics:cache",
              f"prefix={acme.prefix if acme else None}")
    else:
        check("normalize_tenant('Acme') rejected off-list (strict mode)", acme is None,
              f"ctx={acme!r}")

    for raw in (None, "", "UPPER case!", "a:b", "a" * 80, "-leading-dash"):
        check(f"normalize_tenant rejects {raw!r}", normalize_tenant(raw) is None)

    try:
        require_tenant(x_tenant_id="")
        check("require_tenant(missing header) raises 422", False, "no exception raised")
    except HTTPException as exc:
        check("require_tenant(missing header) raises 422", exc.status_code == 422,
              f"status={exc.status_code}")
    except Exception as exc:
        check("require_tenant(missing header) raises 422", False, str(exc)[:200])

    if not TENANTS_OPEN:
        off_list = "off-list-step8-probe"
        while off_list in TENANT_ALLOWLIST:
            off_list += "x"
        try:
            require_tenant(x_tenant_id=off_list)
            check(f"require_tenant(off-list {off_list!r}) raises 403", False,
                  "no exception raised")
        except HTTPException as exc:
            check(f"require_tenant(off-list {off_list!r}) raises 403",
                  exc.status_code == 403, f"status={exc.status_code}")
        except Exception as exc:
            check(f"require_tenant(off-list {off_list!r}) raises 403", False, str(exc)[:200])
    else:
        opened = require_tenant(x_tenant_id="open-mode-step8-probe")
        check("open mode: require_tenant(valid slug) returns context",
              isinstance(opened, TenantContext) and opened.id == "open-mode-step8-probe",
              f"ctx={opened!r}")


def _synthetic_vector() -> np.ndarray:
    """Fixed random float32 vector of EMBEDDING_DIM floats (rng seed 42)."""
    return np.random.default_rng(42).standard_normal(EMBEDDING_DIM).astype(np.float32)


def _num_docs(r: Redis, index: str) -> int:
    def find(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == b"num_docs":
                    return int(v)
                found = find(v)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = find(item)
                if found is not None:
                    return found
        return None

    try:
        return find(r.execute_command("FT.INFO", index)) or 0
    except Exception:
        return 0


def _await_backfill(r: Redis, t: TenantContext, want: int = 1,
                    timeout_s: float = 5.0) -> bool:
    """RediSearch backfills pre-existing hashes into a new index asynchronously."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _num_docs(r, t.doc_index) >= want:
            return True
        time.sleep(0.25)
    return False


def run_redis_drills(r: Redis) -> bool:
    """Part B: two-tenant isolation drills. Returns False when Redis is down."""
    try:
        r.ping()
    except Exception as exc:
        skip("Redis isolation drills",
             f"unreachable at {REDIS_URL}: {str(exc)[:150]}")
        return False

    keys_before = set(r.scan_iter("*"))
    a = TenantContext(TENANT_A)
    b = TenantContext(TENANT_B)
    vec = _synthetic_vector()
    qvec = vec.tobytes()
    chunk_a = {"source_file": "a-doc.asc", "heading": "A", "chunk_pos": 0,
               "chunk_text": "tenant A synthetic corpus chunk", "embedding": vec}
    chunk_b = {"source_file": "b-doc.asc", "heading": "B", "chunk_pos": 0,
               "chunk_text": "tenant B synthetic corpus chunk", "embedding": vec}

    try:
        try:
            idx_a = ingest_mod.create_redis_index(r, a, EMBEDDING_DIM)
            idx_b = ingest_mod.create_redis_index(r, b, EMBEDDING_DIM)
            stored_a = ingest_mod.store_in_redis(r, a, [chunk_a])
            stored_b = ingest_mod.store_in_redis(r, b, [chunk_b])
            ready_a = _await_backfill(r, a)
            ready_b = _await_backfill(r, b)
            check("Drill 1: chunks stored + per-tenant doc indexes created",
                  stored_a == 1 and stored_b == 1 and ready_a and ready_b,
                  f"stored=({stored_a},{stored_b}) "
                  f"idx_created=({idx_a},{idx_b}) num_docs_ready=({ready_a},{ready_b})")
        except Exception as exc:
            check("Drill 1: chunks stored + per-tenant doc indexes created",
                  False, str(exc)[:200])

        try:
            hits_a = retrieval.retrieve("q", r, a, qvec=qvec)
            hits_b = retrieval.retrieve("q", r, b, qvec=qvec)
            files_a = sorted({c["source_file"] for c in hits_a})
            files_b = sorted({c["source_file"] for c in hits_b})
            check("Drill 2: cross-tenant doc isolation",
                  bool(hits_a) and files_a == ["a-doc.asc"]
                  and bool(hits_b) and files_b == ["b-doc.asc"]
                  and not any(c["source_file"] == "b-doc.asc" for c in hits_a),
                  f"A sees={files_a} B sees={files_b}")
        except Exception as exc:
            check("Drill 2: cross-tenant doc isolation", False, str(exc)[:200])

        try:
            cache_mod.ensure_cache_index(r, a)
            cache_mod.ensure_cache_index(r, b)
            stored_key = cache_mod.store(r, a, "step8 cache probe?", qvec, "answer-a", [])
            miss_b = cache_mod.lookup(r, b, qvec)
            hit_a = cache_mod.lookup(r, a, qvec)
            check("Drill 3: cache isolation",
                  bool(stored_key) and miss_b is None
                  and hit_a is not None and hit_a.get("response") == "answer-a",
                  f"stored={bool(stored_key)} B_lookup={'hit' if miss_b else 'None'} "
                  f"A_response={(hit_a or {}).get('response')!r}")
        except Exception as exc:
            check("Drill 3: cache isolation", False, str(exc)[:200])

        try:
            sessions_mod.append(r, a, "s1", "user", "hello-from-A")
            hist_b = sessions_mod.history(r, b, "s1")
            hist_a = sessions_mod.history(r, a, "s1")
            check("Drill 4: session isolation",
                  hist_b == [] and len(hist_a) == 1,
                  f"B_history={len(hist_b)} msgs, A_history={len(hist_a)} msg(s)")
        except Exception as exc:
            check("Drill 4: session isolation", False, str(exc)[:200])

        try:
            req_a0 = metrics_mod.snapshot(r, a)["requests_total"]
            req_b0 = metrics_mod.snapshot(r, b)["requests_total"]
            metrics_mod.record_request(r, a)
            req_a1 = metrics_mod.snapshot(r, a)["requests_total"]
            req_b1 = metrics_mod.snapshot(r, b)["requests_total"]
            check("Drill 5: metrics separation",
                  req_a1 == req_a0 + 1 and req_b1 == req_b0,
                  f"A {req_a0} -> {req_a1}, B {req_b0} -> {req_b1}")
        except Exception as exc:
            check("Drill 5: metrics separation", False, str(exc)[:200])

        try:
            cache_mod.store(r, b, "step8 second B question?", qvec, "answer-b", [])
            count_b_before = cache_mod.count_entries(r, b)
            removed = cache_mod.flush(r, a)
            count_a_after = cache_mod.count_entries(r, a)
            count_b_after = cache_mod.count_entries(r, b)
            check("Drill 6: scoped flush touches only tenant A",
                  count_a_after == 0 and count_b_after == count_b_before and removed >= 1,
                  f"A_removed={removed} A_left={count_a_after} "
                  f"B {count_b_before} -> {count_b_after}")
        except Exception as exc:
            check("Drill 6: scoped flush touches only tenant A", False, str(exc)[:200])

        keys_after = set(r.scan_iter("*"))
        new_keys = keys_after - keys_before
        unscoped = [k.decode(errors="replace") for k in new_keys
                    if not k.startswith(ALLOWED_PREFIXES)]
        legacy = [k.decode(errors="replace") for k in new_keys
                  if k.startswith(LEGACY_PREFIXES)]
        check("Drill 7: key hygiene (zero unprefixed keys created)",
              not unscoped and not legacy,
              f"new_keys={len(new_keys)} unscoped={unscoped[:5]} legacy={legacy[:5]}")
    finally:
        for idx in (a.doc_index, a.cache_index, b.doc_index, b.cache_index):
            try:
                r.execute_command("FT.DROPINDEX", idx)
            except Exception:
                pass
        cleaned = 0
        for pattern in (f"t:{TENANT_A}:*", f"t:{TENANT_B}:*"):
            for key in list(r.scan_iter(match=pattern)):
                try:
                    cleaned += r.delete(key)
                except Exception:
                    pass
        print(f"[CLEANUP] dropped FT indexes + deleted {cleaned} "
              f"t:{TENANT_A}:*/t:{TENANT_B}:* keys")

    return True


def run_api_probes() -> None:
    """Part C: live probes; skipped entirely when /health is unreachable."""
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=2.0)
    except Exception as exc:
        skip("Live API probes", f"{API_BASE}/health unreachable within 2s: {str(exc)[:150]}")
        return

    check("GET /health -> 200", health.status_code == 200, f"status={health.status_code}")

    try:
        no_header = httpx.post(f"{API_BASE}/ask", json={"question": "what can you do?"},
                               timeout=10.0)
        check("POST /ask without X-Tenant-Id -> 422", no_header.status_code == 422,
              f"status={no_header.status_code}")
    except Exception as exc:
        check("POST /ask without X-Tenant-Id -> 422", False, str(exc)[:200])

    headers = {"X-Tenant-Id": "definitely-not-a-tenant"}
    try:
        resp = httpx.post(f"{API_BASE}/ask", json={"question": "hello there"},
                          headers=headers, timeout=60.0)
        if not TENANTS_OPEN:
            check("POST /ask off-list X-Tenant-Id -> 403", resp.status_code == 403,
                  f"status={resp.status_code}")
        else:
            print(f"[INFO] open mode: POST /ask with arbitrary header -> "
                  f"status {resp.status_code} (not asserted)")
    except Exception as exc:
        if not TENANTS_OPEN:
            check("POST /ask off-list X-Tenant-Id -> 403", False, str(exc)[:200])
        else:
            print(f"[INFO] open-mode probe errored (not asserted): {str(exc)[:150]}")


if __name__ == "__main__":
    check_tenancy_units()
    r = Redis.from_url(REDIS_URL, decode_responses=False)
    run_redis_drills(r)
    r.close()
    run_api_probes()
    print(f"\n{'-' * 40}\nStep 8 checks: {PASS}/{PASS + FAIL} passed")
    sys.exit(1 if FAIL else 0)
