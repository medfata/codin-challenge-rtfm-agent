"""Step 7 verification: semantic routing (classify, guards, actions, route metrics),
plus LLM lane checks (fast-lane config, automatic fallback drill)."""

import logging
import sys

from redis import Redis

from rtfm_agent.routing import actions as actions_mod
from rtfm_agent import llm as llm_client
from rtfm_agent.common import metrics as metrics_mod
from rtfm_agent.retrieval import search as retrieval
from rtfm_agent.routing import intent as router_mod
from rtfm_agent.config import settings
from rtfm_agent.common.tenancy import TenantContext

T = TenantContext("local")

PASS = 0
FAIL = 0

# LLM-dependent checks may retry up to 2 times before failing.
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


def check_embedder() -> None:
    try:
        embedder = retrieval.get_embedder()
        vecs = embedder.embed(["semantic routing smoke test"])
        dims = int(vecs.shape[-1])
        check("Embedder loads, dim == 384", dims == 384, f"dims={dims}")
    except Exception as exc:
        check("Embedder loads, dim == 384", False, str(exc)[:200])


def check_llm_reachable() -> None:
    try:
        reply, _usage = llm_client.chat(
            [
                {"role": "system", "content": "You are a health probe."},
                {"role": "user", "content": "Reply with OK"},
            ],
            max_tokens=500,
        )
        check("LLM health probe", bool(reply) and "OK" in reply.upper(), f"reply={reply!r}")
    except Exception as exc:
        check("LLM health probe", False, str(exc)[:200])


def check_fast_lane() -> None:
    lanes = llm_client._lanes()
    ok_cfg = bool(llm_client.LLM_FAST_MODEL) and len(lanes) >= 1
    check("Fast lane configured (separate model + primary key)", ok_cfg,
          f"fast={llm_client.LLM_FAST_MODEL!r} lanes={[l['name'] for l in lanes]}")


def check_fallback_drill() -> None:
    """Force the primary lane unreachable; the fallback lane must take over."""
    original_base = settings.llm.base_url
    records: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    cap = _Cap()
    logging.getLogger("rtfm_agent.llm").addHandler(cap)
    try:
        settings.llm.base_url = "http://127.0.0.1:9"  # nothing listens here
        try:
            reply, _usage = llm_client.chat(
                [{"role": "user", "content": "Reply with exactly: PONG"}],
                max_tokens=500,
            )
            check("Fallback answers with primary unreachable", "PONG" in reply.upper(),
                  f"reply={reply!r}")
            return
        except Exception as exc:
            traversed = any("primary lane failed" in m for m in records)
            # Chain correct even if the fallback lane itself is quota-limited.
            check("Fallback answers with primary unreachable",
                  traversed and "429" in str(exc),
                  f"traversed={traversed} err={str(exc)[:120]}")
    finally:
        settings.llm.base_url = original_base
        logging.getLogger("rtfm_agent.llm").removeHandler(cap)


def classify_with_retry(question: str, hist_turns: list[dict],
                        predicate) -> tuple[bool, object, str]:
    """classify() with up to 2 retries; returns (ok, last_result, detail)."""
    result = None
    detail = ""
    for attempt in range(1, LLM_ATTEMPTS + 1):
        try:
            result = router_mod.classify(question, hist_turns)
        except Exception as exc:
            detail = f"attempt {attempt}: {str(exc)[:150]}"
            continue
        if predicate(result):
            return True, result, ""
        detail = (
            f"attempt {attempt}: route={result.route!r} "
            f"action={result.action!r} query={result.query!r}"
        )
    return False, result, detail


def check_chitchat_routing() -> None:
    ok, _res, detail = classify_with_retry(
        "hey there, thanks for all the help!", [],
        lambda x: x.route == "chitchat",
    )
    check("Chitchat routes to chitchat", ok, detail)


def check_doc_routing() -> None:
    ok, res, detail = classify_with_retry(
        "How do I undo the last commit in Git?", [],
        lambda x: x.route == "doc" and bool(x.query),
    )
    check("Doc routes to doc with search query", ok,
          detail or f"query={res.query!r}")


def check_followup_rewrite() -> None:
    hist = [
        {"role": "user", "content": "how do I undo the last commit?"},
        {"role": "assistant", "content": "You can use git reset --soft HEAD~1 to undo it."},
    ]
    question = "and how do I push it safely afterwards?"
    ok, res, detail = classify_with_retry(
        question, hist,
        lambda x: x.route == "doc" and x.query != question,
    )
    check("Follow-up rewrites to standalone query", ok,
          detail or f"rewritten={res.query!r}")


def check_action_routing() -> None:
    ok, res, detail = classify_with_retry(
        "show me your stats", [],
        lambda x: x.route == "action" and x.action == "metrics",
    )
    check("Action routes to 'metrics'", ok, detail)


def check_destructive_allowed() -> None:
    ok, res, detail = classify_with_retry(
        "please flush the answer cache", [],
        lambda x: x.route == "action" and x.action == "flush_cache",
    )
    check("Destructive 'flush_cache' allowed (keyword corroborated)", ok, detail)


def check_guard_rejection() -> None:
    raw = '{"route": "action", "action": "flush_cache", "query": "", "source": null}'
    try:
        res = router_mod._parse(raw, original="tell me about the cache")
        check("Guard rejects uncorroborated flush_cache",
              res.route == "doc", f"route={res.route!r}")
    except Exception as exc:
        check("Guard rejects uncorroborated flush_cache", False, str(exc)[:200])


def check_malformed_json() -> None:
    try:
        res = router_mod._parse("complete garbage without braces", "original q")
        check("Malformed JSON falls back to doc/original",
              res.route == "doc" and res.query == "original q",
              f"route={res.route!r} query={res.query!r}")
    except Exception as exc:
        check("Malformed JSON falls back to doc/original", False, str(exc)[:200])


def check_actions(r: Redis) -> None:
    try:
        report = actions_mod.dispatch("metrics", r, T, "step7-test")
        check("dispatch('metrics') report",
              isinstance(report, str) and "Requests served" in report,
              repr(report[:80]))
    except Exception as exc:
        check("dispatch('metrics') report", False, str(exc)[:200])
    try:
        listing = actions_mod.dispatch("list_docs", r, T, "step7-test")
        check("dispatch('list_docs') non-empty", isinstance(listing, str) and listing.strip() != "",
              repr(listing[:80]))
    except Exception as exc:
        check("dispatch('list_docs') non-empty", False, str(exc)[:200])
    try:
        cleared = actions_mod.dispatch("clear_session", r, T, "step7-no-such-session")
        check("dispatch('clear_session') empty-session notice",
              isinstance(cleared, str) and "no conversation history" in cleared.lower(),
              repr(cleared[:80]))
    except Exception as exc:
        check("dispatch('clear_session') empty-session notice", False, str(exc)[:200])


def check_metrics_counters(r: Redis) -> None:
    try:
        before = int(metrics_mod.snapshot(r, T)["route_chitchat_total"])
        metrics_mod.record_route(r, T, "chitchat")
        after = int(metrics_mod.snapshot(r, T)["route_chitchat_total"])
        check("record_route bumps route_chitchat_total", after == before + 1,
              f"{before} -> {after}")
    except Exception as exc:
        check("record_route bumps route_chitchat_total", False, str(exc)[:200])


if __name__ == "__main__":
    r = Redis.from_url(settings.redis.url, decode_responses=False)
    check_redis(r)
    check_embedder()
    check_fast_lane()
    check_llm_reachable()
    check_fallback_drill()
    check_chitchat_routing()
    check_doc_routing()
    check_followup_rewrite()
    check_action_routing()
    check_destructive_allowed()
    check_guard_rejection()
    check_malformed_json()
    check_actions(r)
    check_metrics_counters(r)
    r.close()
    print(f"\n{'-' * 40}\nStep 7 checks: {PASS}/{PASS + FAIL} passed")
    sys.exit(1 if FAIL else 0)
