"""Observability metrics persisted in Redis (INCR-based, survive restarts).

Tracks cache efficiency, request latency, token usage, and estimated cost.
Each tenant's counters live under its own hash at {prefix}metrics:cache.
"""

import logging

from redis import Redis

from rtfm_agent.common.tenancy import TenantContext
from rtfm_agent.config import settings

logger = logging.getLogger(__name__)


def record_hit(r: Redis, t: TenantContext, latency_ms: float) -> None:
    try:
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(t.metrics_key, "cache_hits", 1)
        pipe.hincrbyfloat(t.metrics_key, "cached_latency_ms_total", latency_ms)
        pipe.execute()
    except Exception as exc:
        logger.warning("metrics write failed: %s", exc)


def record_miss(r: Redis, t: TenantContext, latency_ms: float) -> None:
    try:
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(t.metrics_key, "cache_misses", 1)
        pipe.hincrbyfloat(t.metrics_key, "uncached_latency_ms_total", latency_ms)
        pipe.execute()
    except Exception as exc:
        logger.warning("metrics write failed: %s", exc)


def record_request(r: Redis, t: TenantContext, prompt_tokens: int = 0,
                   completion_tokens: int = 0, errored: bool = False) -> None:
    """Count a request and accumulate token usage / cost estimate."""
    try:
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(t.metrics_key, "requests_total", 1)
        if errored:
            pipe.hincrby(t.metrics_key, "errors_total", 1)
        if prompt_tokens:
            pipe.hincrbyfloat(t.metrics_key, "prompt_tokens_total", prompt_tokens)
            cost = prompt_tokens / 1e6 * settings.pricing.input_per_mtok
            pipe.hincrbyfloat(t.metrics_key, "estimated_cost_usd_total", cost)
        if completion_tokens:
            pipe.hincrbyfloat(t.metrics_key, "completion_tokens_total", completion_tokens)
            cost = completion_tokens / 1e6 * settings.pricing.output_per_mtok
            pipe.hincrbyfloat(t.metrics_key, "estimated_cost_usd_total", cost)
        pipe.execute()
    except Exception as exc:
        logger.warning("metrics write failed: %s", exc)


def record_route(r: Redis, t: TenantContext, route: str) -> None:
    """Count one routed message (doc/chitchat/action/memory)."""
    try:
        r.hincrby(t.metrics_key, f"route_{route}_total", 1)
    except Exception as exc:
        logger.warning("metrics write failed: %s", exc)


def record_lane_call(r: Redis, t: TenantContext, lane: str) -> None:
    """Count one successful LLM call on a named lane (generation/fast/economy)."""
    try:
        r.hincrby(t.metrics_key, f"llm_calls_{lane}_total", 1)
    except Exception as exc:
        logger.warning("metrics write failed: %s", exc)


def record_cache_warm(r: Redis, t: TenantContext, answers: int) -> None:
    """Count one cache-warm run and the answers it wrote."""
    try:
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(t.metrics_key, "cache_warm_runs_total", 1)
        if answers:
            pipe.hincrby(t.metrics_key, "cache_warm_answers_total", answers)
        pipe.execute()
    except Exception as exc:
        logger.warning("metrics write failed: %s", exc)


def record_stale_answer(r: Redis, t: TenantContext) -> None:
    """Count an answer served with an outdated-documentation warning."""
    try:
        r.hincrby(t.metrics_key, "stale_answers_served", 1)
    except Exception as exc:
        logger.warning("metrics write failed: %s", exc)


def record_mcp_call(r: Redis, t: TenantContext) -> None:
    """Count one MCP tool/resource invocation for this tenant."""
    try:
        r.hincrby(t.metrics_key, "mcp_calls_total", 1)
    except Exception as exc:
        logger.warning("metrics write failed: %s", exc)


def record_crawl(r: Redis, t: TenantContext, pages_fetched: int = 0,
                 failures: int = 0, discarded: bool = False) -> None:
    """Count one finished crawl job plus its fetch/failure volumes."""
    try:
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(t.metrics_key, "crawl_jobs_total", 1)
        if pages_fetched:
            pipe.hincrby(t.metrics_key, "crawl_pages_fetched_total", pages_fetched)
        if failures:
            pipe.hincrby(t.metrics_key, "crawl_failures_total", failures)
        if discarded:
            pipe.hincrby(t.metrics_key, "crawl_discarded_total", 1)
        pipe.execute()
    except Exception as exc:
        logger.warning("metrics write failed: %s", exc)


def snapshot(r: Redis, t: TenantContext) -> dict:
    data = r.hgetall(t.metrics_key)

    def num(field: str) -> float:
        v = data.get(field.encode(), b"0")
        return float(v) if v else 0.0

    hits = int(num("cache_hits"))
    misses = int(num("cache_misses"))
    total = hits + misses
    requests = int(num("requests_total"))
    return {
        "requests_total": requests,
        "errors_total": int(num("errors_total")),
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "avg_cached_ms": round(num("cached_latency_ms_total") / hits, 1) if hits else 0.0,
        "avg_uncached_ms": round(num("uncached_latency_ms_total") / misses, 1) if misses else 0.0,
        "total_questions": total,
        "prompt_tokens_total": int(num("prompt_tokens_total")),
        "completion_tokens_total": int(num("completion_tokens_total")),
        "estimated_cost_usd": round(num("estimated_cost_usd_total"), 6),
        "route_doc_total": int(num("route_doc_total")),
        "route_chitchat_total": int(num("route_chitchat_total")),
        "route_action_total": int(num("route_action_total")),
        "route_memory_total": int(num("route_memory_total")),
        "stale_answers_served": int(num("stale_answers_served")),
        "mcp_calls_total": int(num("mcp_calls_total")),
        "events_published_total": int(num("events_published_total")),
        "llm_calls_generation_total": int(num("llm_calls_generation_total")),
        "llm_calls_fast_total": int(num("llm_calls_fast_total")),
        "llm_calls_economy_total": int(num("llm_calls_economy_total")),
        "cache_warm_runs_total": int(num("cache_warm_runs_total")),
        "cache_warm_answers_total": int(num("cache_warm_answers_total")),
        "crawl_jobs_total": int(num("crawl_jobs_total")),
        "crawl_pages_fetched_total": int(num("crawl_pages_fetched_total")),
        "crawl_failures_total": int(num("crawl_failures_total")),
        "crawl_discarded_total": int(num("crawl_discarded_total")),
    }
