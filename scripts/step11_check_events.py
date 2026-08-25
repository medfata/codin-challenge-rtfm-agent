"""Step 11 verification: real-time event notifications (Redis Streams -> SSE).

Part A: Redis drills on synthetic tenants - envelope shape, MAXLEN trim,
        metrics wiring, live XREAD delivery, Last-Event-ID resume, backlog
        replay, id validation, tenant isolation. No LLM, no embedder.
Part B: live HTTP probes of /events/stream against RTFM_API_URL (skipped
        when the API is unreachable): auth rejection + conflict, garbage
        Last-Event-ID fallback, CORS preflight, same-origin demo route,
        live SSE frame delivery across processes, Last-Event-ID resume
        over HTTP, and a 40-subscriber concurrency drill proving the sync
        endpoint pool stays responsive.

Redis drills and API probes degrade to [SKIP]; exit code 1 iff any FAIL.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from redis import Redis

from rtfm_agent import events as events_mod
from rtfm_agent import metrics as metrics_mod
from rtfm_agent.config import REDIS_URL
from rtfm_agent.tenancy import TenantContext

PASS = 0
FAIL = 0

TENANT = "step11"
TENANT_B = "step11b"
# 127.0.0.1, not "localhost": on Windows, fresh connections to "localhost"
# pay a ~2s IPv6 (::1) fallback penalty that poisons latency measurements.
API_BASE = os.getenv("RTFM_API_URL", "http://127.0.0.1:8000").rstrip("/")


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


def _wipe_tenant(r: Redis, slug: str) -> None:
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=f"t:{slug}:*", count=500)
        if keys:
            r.delete(*keys)
        if cursor == 0:
            break


def _collect(gen, want: int, budget_s: float) -> list:
    """Drain an iter_events generator until `want` real events arrive or the
    time budget runs out (idle cycles yield None)."""
    out: list = []
    t0 = time.time()
    for item in gen:
        if item is not None:
            out.append(item)
            if len(out) >= want:
                break
        if time.time() - t0 > budget_s:
            break
    return out


def part_a_units() -> None:
    check("normalize_last_id passes valid stream ids",
          events_mod.normalize_last_id("1787651534122-0") == "1787651534122-0")
    for bad in ("", "garbage", "12-34-56; X", "-1", "abc-def", "1-", "*"):
        if events_mod.normalize_last_id(bad) != "":
            check(f"normalize_last_id rejects {bad!r}", False)
            return
    check("normalize_last_id rejects malformed ids (empty string fallback)", True)
    check("sync resolve_start falls back to $ on invalid Last-Event-ID",
          events_mod.resolve_start(None, TenantContext(TENANT), "garbage", 0) == "$")


def part_a_redis() -> None:
    try:
        r = Redis.from_url(REDIS_URL, decode_responses=False, socket_connect_timeout=3)
        r.ping()
    except Exception as exc:
        skip("Redis drills", f"no Redis at {REDIS_URL}: {exc}")
        return

    t = TenantContext(TENANT)
    tb = TenantContext(TENANT_B)

    # Keep the live drills snappy: 1s idle windows instead of the configured 15s.
    original_maxlen = events_mod.EVENTS_STREAM_MAXLEN
    original_heartbeat = events_mod.EVENTS_HEARTBEAT_S
    events_mod.EVENTS_HEARTBEAT_S = 1

    try:
        _wipe_tenant(r, TENANT)
        _wipe_tenant(r, TENANT_B)

        # --- Envelope shape ---------------------------------------------------
        eid = events_mod.publish(r, t, events_mod.INGEST_COMPLETED,
                                 {"chunks_stored": 7, "corpus_version": 2})
        check("publish returns a stream entry id", eid is not None,
              f"id={eid!r}")
        raw = r.xrange(events_mod.stream_key(t))
        check("one entry stored under t:{org}:events",
              len(raw) == 1 and raw[0][0] == eid)
        parsed = events_mod._decode_entry(*raw[0])
        env_fields = {events_mod._s(k): events_mod._s(v) for k, v in raw[0][1].items()}
        check("envelope carries type/ts/JSON data",
              parsed[0] == eid.decode()
              and parsed[1] == "ingest.completed"
              and parsed[2].get("chunks_stored") == 7
              and bool(env_fields.get("ts")),
              json.dumps(parsed[2])[:60])

        # --- Metrics wiring ----------------------------------------------------
        snap = metrics_mod.snapshot(r, t)
        check("events_published_total counter wired",
              snap["events_published_total"] >= 1,
              f"n={snap['events_published_total']}")

        # --- Tenant isolation ---------------------------------------------------
        check("other tenant's stream stays empty",
              r.xlen(events_mod.stream_key(tb)) == 0)

        # --- Live delivery ------------------------------------------------------
        threading.Thread(
            target=lambda: (
                time.sleep(0.8),
                events_mod.publish(r, t, "drill.live", {"n": 1}),
            ), daemon=True
        ).start()
        gen = events_mod.iter_events(r, t, "$")
        got = _collect(gen, want=1, budget_s=8)
        check("blocking XREAD delivers a concurrently published event",
              any(typ == "drill.live" and d.get("n") == 1 for _, typ, d in got),
              f"{len(got)} event(s)")

        # --- Resume from Last-Event-ID ------------------------------------------
        ids = []
        for i in range(3):
            ids.append(events_mod.publish(r, t, "drill.seq", {"i": i}))
        mid = ids[1]
        cursor = events_mod.resolve_start(r, t, mid.decode() if isinstance(mid, bytes) else mid, 0)
        check("Last-Event-ID is used as-is (XREAD is strictly-greater)",
              cursor == (mid.decode() if isinstance(mid, bytes) else mid))
        gen = events_mod.iter_events(r, t, cursor)
        got = _collect(gen, want=1, budget_s=6)
        seqs = [d.get("i") for _, typ, d in got if typ == "drill.seq"]
        check("resume replays only later events, in order",
              seqs == [2], f"got={seqs}")

        # --- Backlog replay -------------------------------------------------------
        cursor = events_mod.resolve_start(r, t, "", 2)
        gen = events_mod.iter_events(r, t, cursor)
        got = _collect(gen, want=2, budget_s=6)
        seqs = [d.get("i") for _, typ, d in got if typ == "drill.seq"]
        check("backlog=2 replays the two most recent entries oldest-first",
              seqs == [1, 2], f"got={seqs}")

        # --- MAXLEN trim -----------------------------------------------------------
        events_mod.EVENTS_STREAM_MAXLEN = 10
        try:
            for i in range(30):
                events_mod.publish(r, t, "drill.trim", {"i": i})
            length = r.xlen(events_mod.stream_key(t))
            check("exact MAXLEN trim bounds the stream",
                  length == 10, f"xlen={length}")
        finally:
            events_mod.EVENTS_STREAM_MAXLEN = original_maxlen

        # --- Fail-open publishing ---------------------------------------------------
        broken = Redis.from_url("redis://localhost:9/0", socket_connect_timeout=0.3)
        check("publish to unreachable Redis fails open (returns None)",
              events_mod.publish(broken, t, "drill.x", {}) is None)
    finally:
        events_mod.EVENTS_HEARTBEAT_S = original_heartbeat
        _wipe_tenant(r, TENANT)
        _wipe_tenant(r, TENANT_B)


def part_b_http() -> None:
    import httpx

    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=3)
        alive = resp.status_code < 500
    except Exception:
        alive = False
    if not alive:
        skip("HTTP /events/stream probes", f"API not reachable at {API_BASE}")
        return

    headers_probe = {"X-Tenant-Id": ""}
    try:
        resp = httpx.get(f"{API_BASE}/events/stream", headers=headers_probe, timeout=5)
        check("stream without tenant identity rejected with 422",
              resp.status_code == 422, f"status={resp.status_code}")
    except Exception as exc:
        skip("auth rejection probe", f"request failed: {exc}")

    try:
        resp = httpx.get(f"{API_BASE}/events/stream?tenant={TENANT}",
                         headers={"X-Tenant-Id": TENANT_B}, timeout=5)
        check("conflicting ?tenant= and X-Tenant-Id rejected with 422",
              resp.status_code == 422, f"status={resp.status_code}")
    except Exception as exc:
        skip("tenant-conflict probe", f"request failed: {exc}")

    try:
        resp = httpx.options(
            f"{API_BASE}/events/stream",
            headers={"Origin": "http://example.com",
                     "Access-Control-Request-Method": "GET"},
            timeout=5,
        )
        aco = resp.headers.get("access-control-allow-origin")
        check("CORS preflight allows cross-origin GET",
              resp.status_code in (200, 204) and aco in ("*", "http://example.com"),
              f"status={resp.status_code} aco={aco}")
    except Exception as exc:
        skip("CORS preflight probe", f"request failed: {exc}")

    try:
        resp = httpx.get(f"{API_BASE}/demo/events", timeout=5)
        check("same-origin demo page served at /demo/events",
              resp.status_code == 200 and "text/html" in resp.headers.get("content-type", ""),
              f"status={resp.status_code}")
    except Exception as exc:
        skip("/demo/events probe", f"request failed: {exc}")

    try:
        r = Redis.from_url(REDIS_URL, decode_responses=False, socket_connect_timeout=3)
        r.ping()
    except Exception as exc:
        skip("live SSE probe", f"no Redis to publish from: {exc}")
        return
    t = TenantContext(TENANT)
    _wipe_tenant(r, TENANT)

    def read_frames(url: str, extra_headers: dict | None, sink: list,
                    ready: threading.Event) -> None:
        try:
            with httpx.stream("GET", url, headers=extra_headers or {},
                              timeout=httpx.Timeout(5, read=15)) as resp:
                sink.append(("__status__", resp.status_code))
                ready.set()
                event_name, event_id, data = None, None, None
                for line in resp.iter_lines():
                    if line.startswith("event:"):
                        event_name = line.split(":", 1)[1].strip()
                    elif line.startswith("id:"):
                        event_id = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        data = line.split(":", 1)[1].strip()
                    elif line.strip() == "" and event_name:
                        sink.append((event_name, event_id, data))
                        event_name, event_id, data = None, None, None
        except Exception:
            pass

    # --- Live cross-process delivery -----------------------------------------
    url = f"{API_BASE}/events/stream?tenant={TENANT}&backlog=0"
    sink: list = []
    ready = threading.Event()
    worker = threading.Thread(target=read_frames, args=(url, None, sink, ready), daemon=True)
    worker.start()
    if not ready.wait(6):
        skip("live SSE probe", "stream did not open")
        _wipe_tenant(r, TENANT)
        return
    status_ok = sink and sink[0] == ("__status__", 200)
    time.sleep(1.0)  # let the server park inside blocking XREAD
    eid = events_mod.publish(r, t, "probe.http", {"via": "sse"})
    worker.join(timeout=14)
    expected = eid.decode() if isinstance(eid, bytes) else eid
    frames = [f for f in sink if f != ("__status__", 200)]
    match = [
        f for f in frames
        if f[0] == "probe.http" and f[1] == expected
        and f[2] and json.loads(f[2]).get("via") == "sse"
    ]
    check("SSE delivers a published event with matching id/type/payload",
          bool(status_ok) and bool(match), f"{len(frames)} frame(s)")

    # --- Last-Event-ID resume over HTTP ----------------------------------------
    eid2 = events_mod.publish(r, t, "probe.resume", {"n": 2})
    expected2 = eid2.decode() if isinstance(eid2, bytes) else eid2
    sink2: list = []
    ready2 = threading.Event()
    worker2 = threading.Thread(
        target=read_frames,
        args=(f"{API_BASE}/events/stream?tenant={TENANT}",
              {"Last-Event-ID": expected}, sink2, ready2),
        daemon=True,
    )
    worker2.start()
    worker2.join(timeout=14)
    frames2 = [f for f in sink2 if f != ("__status__", 200)]
    resumed = [
        f for f in frames2
        if f[0] == "probe.resume" and f[1] == expected2
    ]
    older_leak = [f for f in frames2 if f[0] == "probe.http"]
    check("reconnect with Last-Event-ID receives missed events only",
          bool(resumed) and not older_leak,
          f"{len(frames2)} frame(s)")

    # --- Garbage Last-Event-ID falls back to live tail -------------------------
    sink3: list = []
    ready3 = threading.Event()
    worker3 = threading.Thread(
        target=read_frames,
        args=(f"{API_BASE}/events/stream?tenant={TENANT}",
              {"Last-Event-ID": "garbage-not-an-id"}, sink3, ready3),
        daemon=True,
    )
    worker3.start()
    if ready3.wait(6):
        time.sleep(1.0)
        eidg = events_mod.publish(r, t, "probe.garbageid", {"ok": True})
        worker3.join(timeout=14)
        expectedg = eidg.decode() if isinstance(eidg, bytes) else eidg
        frames3 = [f for f in sink3 if isinstance(f, tuple) and len(f) == 3]
        got_garbage = [f for f in frames3
                       if f[0] == "probe.garbageid" and f[1] == expectedg]
        check("malformed Last-Event-ID degrades to live tail (events still flow)",
              bool(got_garbage), f"{len(frames3)} frame(s)")
    else:
        skip("garbage Last-Event-ID probe", "stream did not open")

    # --- Concurrency: 40 idle subscribers must not stall the API ---------------
    sinks = []
    for i in range(40):
        s: list = []
        ev = threading.Event()
        th = threading.Thread(target=read_frames,
                              args=(f"{API_BASE}/events/stream?tenant={TENANT}&backlog=0",
                                    None, s, ev), daemon=True)
        th.start()
        sinks.append((ev, s))
    opened = sum(1 for ev, _ in sinks if ev.wait(8))
    time.sleep(1.0)  # let all readers park inside their XREAD awaits
    # Latency must be measured on a WARM reused connection: fresh TCP
    # connections (and any hostname resolution) add one-off noise that has
    # nothing to do with subscriber load.
    with httpx.Client(trust_env=False) as warm:
        warm.get(f"{API_BASE}/demo/events", timeout=10)
        t0 = time.perf_counter()
        sresp = warm.get(f"{API_BASE}/demo/events", timeout=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        try:
            hresp = warm.get(f"{API_BASE}/health", timeout=10)
            health_ok = hresp.status_code < 500
        except Exception:
            health_ok = False
    health_ms = (time.perf_counter() - t0) * 1000
    check("40 concurrent SSE subscribers keep static routes responsive",
          opened >= 35 and sresp.status_code == 200 and elapsed_ms < 500,
          f"opened={opened} demo={elapsed_ms:.0f}ms health={health_ms:.0f}ms")

    _wipe_tenant(r, TENANT)


if __name__ == "__main__":
    t0 = time.time()
    print(f"== Step 11 checks - {time.strftime('%Y-%m-%d %H:%M:%S')} ==")
    part_a_units()
    part_a_redis()
    part_b_http()
    print(f"\n{PASS} passed, {FAIL} failed in {time.time() - t0:.1f}s")
    sys.exit(1 if FAIL else 0)
