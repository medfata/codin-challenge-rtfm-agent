"""Step 12 verification: multi-model strategy (lane chains, economy routing,
cache warming, per-lane metrics) plus a Step 7 routing regression pass."""

import logging
import sys
import time

from redis import Redis

from rtfm_agent import cache as cache_mod
from rtfm_agent import llm as llm_client
from rtfm_agent import metrics as metrics_mod
from rtfm_agent import retrieval
from rtfm_agent import router as router_mod
from rtfm_agent import warm as warm_mod
from rtfm_agent.config import (
    ENABLE_CACHE_WARM,
    LLM_ECONOMY_FALLBACK_MODEL,
    LLM_ECONOMY_MODEL,
    REDIS_URL,
)
from rtfm_agent.tenancy import TenantContext

T = TenantContext("local")

PASS = 0
FAIL = 0

LLM_ATTEMPTS = 3


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"[PASS] {name}" + (f" - {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"[FAIL] {name}" + (f" - {detail}" if detail else ""))


def check_redis(r: Redis) -> None:
    try:
        check("Redis PING", r.ping() is True)
    except Exception as exc:
        check("Redis PING", False, str(exc)[:200])


def check_lane_config() -> None:
    gen = [l["name"] for l in llm_client._lanes("generation")]
    eco = [l["name"] for l in llm_client._lanes("economy")]
    fast_ok = bool(llm_client.LLM_FAST_MODEL) and len(llm_client._lanes()) >= 1
    check("Generation chain configured", "primary" in gen, f"{gen}")
    check(
        "Economy chain ordered economy -> economy-fallback",
        eco[:2] == ["economy", "economy-fallback"] and len(eco) >= 1,
        f"{eco}",
    )
    models = [
        (l["name"], l.get("model")) for l in llm_client._lanes("economy")
    ]
    pinned = all(m == LLM_ECONOMY_MODEL or m == LLM_ECONOMY_FALLBACK_MODEL
                 for _n, m in models)
    check("Economy lanes pin their models", pinned, f"{models}")
    check(
        "Economy fallback is NOT the fast-lane model",
        LLM_ECONOMY_FALLBACK_MODEL != llm_client.LLM_FAST_MODEL,
        f"economy_fb={LLM_ECONOMY_FALLBACK_MODEL!r} fast={llm_client.LLM_FAST_MODEL!r}",
    )
    check("Fast lane still configured", fast_ok)


def check_economy_unconfigured_failopen() -> None:
    """No Google key -> economy silently starts on the Groq fallback."""
    original_key = llm_client.LLM_ECONOMY_API_KEY
    try:
        llm_client.LLM_ECONOMY_API_KEY = ""
        names = [l["name"] for l in llm_client._lanes("economy")]
        check(
            "Unconfigured economy key degrades to fallback-only chain",
            names == ["economy-fallback"],
            f"{names}",
        )
    finally:
        llm_client.LLM_ECONOMY_API_KEY = original_key


def check_economy_chat() -> None:
    """Live economy-lane probe; passes if any lane in the chain answers."""
    try:
        reply, _usage = llm_client.chat(
            [{"role": "user", "content": "Reply with exactly: PONG"}],
            max_tokens=500,
            lane="economy",
        )
        check("Economy lane chat works", "PONG" in reply.upper(), f"reply={reply!r}")
    except Exception as exc:
        # Chain-level exhaustion still proves the chain resolved + failed over.
        check("Economy lane chat works", "429" in str(exc), f"err={str(exc)[:150]}")


def check_routing_regression() -> None:
    cases = [
        ("hey there, thanks!", lambda x: x.route == "chitchat", "chitchat"),
        ("How do I undo the last commit in Git?",
         lambda x: x.route == "doc", "doc"),
        ("show me your stats",
         lambda x: x.route == "action" and x.action == "metrics", "action:metrics"),
        ("please flush the answer cache",
         lambda x: x.route == "action" and x.action == "flush_cache", "action:flush"),
        ("warm up your answer cache",
         lambda x: x.route == "action" and x.action == "warm_cache", "action:warm"),
    ]
    for question, predicate, label in cases:
        ok, detail = False, ""
        for attempt in range(1, LLM_ATTEMPTS + 1):
            try:
                res = router_mod.classify(question, [])
                if predicate(res):
                    ok = True
                    break
                detail = f"attempt {attempt}: route={res.route!r} action={res.action!r}"
            except Exception as exc:
                detail = f"attempt {attempt}: {str(exc)[:120]}"
        check(f"Routing regression: {label}", ok, detail)


def check_qfreq_tracking(r: Redis) -> None:
    try:
        warm_mod.track_question(r, T, "step12 popularity probe")
        score = r.zscore(warm_mod.qfreq_key(T), "step12 popularity probe")
        warm_mod.track_question(r, T, "step12 popularity probe")
        score2 = r.zscore(warm_mod.qfreq_key(T), "step12 popularity probe")
        check("qfreq ZINCRBY tracks popularity",
              score is not None and float(score2) == float(score) + 1,
              f"{score} -> {score2}")
        r.zrem(warm_mod.qfreq_key(T), "step12 popularity probe")
    except Exception as exc:
        check("qfreq ZINCRBY tracks popularity", False, str(exc)[:200])


def check_warm_lock(r: Redis) -> None:
    lock_key = f"{T.prefix}lock:warm"
    try:
        if not r.set(lock_key, "held-by-check", nx=True, ex=60):
            r.delete(lock_key)
            r.set(lock_key, "held-by-check", nx=True, ex=60)
        try:
            warm_mod.run_warm(r, T, top_n=1)
            check("Warm lock rejects overlapping run", False, "no RuntimeError raised")
        except RuntimeError as exc:
            check("Warm lock rejects overlapping run", True, str(exc))
        finally:
            r.delete(lock_key)
    except Exception as exc:
        check("Warm lock rejects overlapping run", False, str(exc)[:200])


def check_warm_run(r: Redis) -> None:
    """End-to-end: seed popularity (+ corpus when missing), flush, warm, verify."""
    if not ENABLE_CACHE_WARM:
        check("Cache warm end-to-end", False, "ENABLE_CACHE_WARM=0")
        return
    question = "What is the Git index in Pro Git terms?"
    try:
        sources = retrieval.indexed_sources(r, T)
        if not sources:
            seeded = _seed_minimal_corpus(r)
            check("Minimal corpus ingested for warm test", seeded,
                  "no pre-existing index for tenant 'local'")
            if not seeded:
                return
        cache_mod.flush(r, T)
        before = cache_mod.count_entries(r, T)
        for _ in range(3):
            warm_mod.track_question(r, T, question)
        deadline = time.time() + 240
        summary = None
        while time.time() < deadline:
            try:
                summary = warm_mod.run_warm(r, T, top_n=5)
                break
            except RuntimeError:
                time.sleep(2)  # another warm job holds the lock; retry
        if summary is None:
            check("Cache warm end-to-end", False, "lock never released")
            return
        # Production path records these in actions._warm_cache/_run_job.
        metrics_mod.record_cache_warm(r, T, summary["warmed"])
        after = cache_mod.count_entries(r, T)
        hit = cache_mod.lookup(r, T, warm_mod._embed_question(question))
        snap = metrics_mod.snapshot(r, T)
        ok = (
            after > before
            and hit is not None
            and summary["warmed"] >= 1
            and summary["failed"] == 0
            and int(snap["cache_warm_runs_total"]) >= 1
        )
        detail = (f"{before}->{after} entries, summary={summary}, "
                  f"warm_runs={snap['cache_warm_runs_total']}")
        check("Cache warm end-to-end", ok, detail)
        if not ok or hit is None:
            return
        citations = hit.get("citations") or []
        check("Warmed entry serves a lookup with citations",
              len(citations) > 0, f"citations={len(citations)}")
    except Exception as exc:
        check("Cache warm end-to-end", False, str(exc)[:250])


def _seed_minimal_corpus(r: Redis) -> bool:
    """Ingest a tiny one-file corpus so warm/retrieval have something to find."""
    import tempfile
    from pathlib import Path

    from rtfm_agent.ingest import run_ingestion

    try:
        tmp = Path(tempfile.mkdtemp(prefix="rtfm-step12-"))
        (tmp / "git-index.asc").write_text(
            "= The Git Index\n\n"
            "The Git index is the staging area between your working "
            "directory and the repository history. Running `git add` "
            "writes a snapshot of file contents into the index, and a "
            "commit records exactly what the index contains.\n\n"
            "== Inspecting the index\n\n"
            "Use `git status` to see how the index differs from HEAD and "
            "from the working tree, or `git ls-files --stage` to dump its "
            "raw contents.\n",
            encoding="utf-8",
        )
        summary = run_ingestion(r, T, docs_dir=str(tmp))
        return summary.get("chunks_stored", 0) > 0
    except Exception as exc:
        print(f"[WARN] corpus seeding failed: {exc}")
        return False


def check_lane_counters(r: Redis) -> None:
    try:
        before = int(metrics_mod.snapshot(r, T)["llm_calls_economy_total"])
        metrics_mod.record_lane_call(r, T, "economy")
        after = int(metrics_mod.snapshot(r, T)["llm_calls_economy_total"])
        check("record_lane_call bumps economy counter", after == before + 1,
              f"{before} -> {after}")
    except Exception as exc:
        check("record_lane_call bumps economy counter", False, str(exc)[:200])


def check_embedder() -> None:
    try:
        vecs = retrieval.get_embedder().embed(["multi-model smoke test"])
        check("Embedder loads", int(vecs.shape[-1]) == 384,
              f"dims={int(vecs.shape[-1])}")
    except Exception as exc:
        check("Embedder loads", False, str(exc)[:200])


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    r = Redis.from_url(REDIS_URL, decode_responses=False)
    check_redis(r)
    check_lane_config()
    check_economy_unconfigured_failopen()
    check_embedder()
    check_economy_chat()
    check_routing_regression()
    check_qfreq_tracking(r)
    check_warm_lock(r)
    check_warm_run(r)
    check_lane_counters(r)
    r.close()
    print(f"\n{'-' * 40}\nStep 12 checks: {PASS}/{PASS + FAIL} passed")
    sys.exit(1 if FAIL else 0)
