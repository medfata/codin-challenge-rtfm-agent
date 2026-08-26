"""Conversational action handlers for the action route.

Deterministic, no LLM: the router picks the action, these functions do the
work and return a plain-text reply the API can serve directly.

Destructive operations (flush_cache, reingest) are additionally gated by
routing.destructive_enabled; the router has already corroborated the intent
with a keyword regex over the raw question before dispatch.
"""

import logging
import threading
import time

from redis import Redis

from rtfm_agent.common.events import INGEST_FAILED, publish
from rtfm_agent.common.metrics import snapshot
from rtfm_agent.common.paths import docs_dir_for
from rtfm_agent.common.sessions import clear as clear_session
from rtfm_agent.common.tenancy import TenantContext
from rtfm_agent.config import settings
from rtfm_agent.ingestion import versioning
from rtfm_agent.retrieval import cache as cache_mod
from rtfm_agent.retrieval.search import indexed_sources

logger = logging.getLogger(__name__)


def dispatch(action: str, r: Redis, t: TenantContext, session_id: str) -> str:
    """Execute a routed action and return the reply text.

    Fail-open: unknown or failing actions return a short notice instead of
    raising - the caller never falls over because of a handler bug.
    """
    if not settings.routing.actions_enabled:
        return "Actions are disabled in this deployment (ENABLE_ACTIONS=0)."

    handlers = {
        "metrics": _metrics_report,
        "list_docs": _list_documents,
        "clear_session": _clear_session,
        "flush_cache": _flush_cache,
        "reingest": _reingest,
        "warm_cache": _warm_cache,
    }
    handler = handlers.get(action)
    if handler is None:
        return f"I don't know how to perform '{action}'."
    try:
        return handler(r, t, session_id)
    except Exception as exc:
        logger.warning("action '%s' failed: %s", action, exc)
        return f"The '{action}' action failed: {exc}"


def _metrics_report(r: Redis, t: TenantContext, session_id: str) -> str:
    """This tenant's counters only - never a cross-tenant aggregate."""
    snap = snapshot(r, t)
    lines = [
        "Current service stats:",
        f"- Requests served: {snap['requests_total']} ({snap['errors_total']} errors); "
        f"cache-tracked doc questions: {snap['total_questions']}",
        f"- Semantic cache: {snap['cache_hits']} hits / {snap['cache_misses']} misses "
        f"(hit rate {snap['hit_rate']:.0%}), size {cache_mod.count_entries(r, t)}",
        f"- Avg latency: {snap['avg_cached_ms']:.0f} ms cached / "
        f"{snap['avg_uncached_ms']:.0f} ms uncached",
        f"- Tokens: {snap['prompt_tokens_total']} in / "
        f"{snap['completion_tokens_total']} out "
        f"(est. cost ${snap['estimated_cost_usd']:.4f})",
        f"- Stale-documentation answers served: {snap.get('stale_answers_served', 0)}",
        f"- MCP tool calls: {snap.get('mcp_calls_total', 0)}",
        f"- Web crawls: {snap.get('crawl_jobs_total', 0)} jobs, "
        f"{snap.get('crawl_pages_fetched_total', 0)} pages fetched "
        f"({snap.get('crawl_failures_total', 0)} failures)",
    ]
    routes = (
        f"doc {int(snap.get('route_doc_total', 0))}, "
        f"chitchat {int(snap.get('route_chitchat_total', 0))}, "
        f"action {int(snap.get('route_action_total', 0))}, "
        f"memory {int(snap.get('route_memory_total', 0))}"
    )
    lines.append(f"- Routes seen: {routes}")
    return "\n".join(lines)


def _list_documents(r: Redis, t: TenantContext, session_id: str) -> str:
    """List the calling tenant's indexed sources (its own doc index only)."""
    entries = indexed_sources(r, t)
    if not entries:
        return (
            "No documents are indexed yet. Trigger ingestion "
            "(POST /ingest) or ask me to re-index the docs."
        )
    total_chunks = sum(e["chunks"] for e in entries)
    lines = [f"{len(entries)} documents indexed ({total_chunks} chunks):"]
    lines += [f"- {e['source_file']} ({e['chunks']} chunks)" for e in entries]

    if settings.versioning.enabled:
        corpus = versioning.get_corpus(r, t)
        if corpus:
            stamp = time.strftime(
                "%Y-%m-%d %H:%M UTC", time.gmtime(corpus["ingested_at"])
            )
            lines.insert(0, f"Corpus version {corpus['version']} "
                            f"(ingested {stamp}):")
        if settings.versioning.drift_warnings:
            try:
                drift = versioning.scan_drift(r, t, docs_dir_for(t))
                changed = versioning.drift_changed_count(drift)
                if changed:
                    lines.append(
                        f"Note: {changed} documents have changed on disk since "
                        f"this ingestion - re-ingest to pick up the new content."
                    )
            except Exception as exc:
                logger.warning("list_docs drift check failed: %s", exc)
    return "\n".join(lines)


def _clear_session(r: Redis, t: TenantContext, session_id: str) -> str:
    """Clear this tenant's session history (t:{org}:session:{sid})."""
    removed = clear_session(r, t, session_id)
    if removed:
        return "Done - this conversation is cleared. Fresh start whenever you're ready."
    return "There was no conversation history to clear."


def _flush_cache(r: Redis, t: TenantContext, session_id: str) -> str:
    """Clear THIS tenant's cache entries only (t:{org}:cache:*)."""
    if not settings.routing.destructive_enabled:
        return "Flushing the cache is disabled (ENABLE_DESTRUCTIVE_ACTIONS=0)."
    removed = cache_mod.flush(r, t)
    noun = "entry" if removed == 1 else "entries"
    return f"Semantic cache flushed - {removed} {noun} removed."


def _reingest(r: Redis, t: TenantContext, session_id: str) -> str:
    """Re-index this tenant's docs corpus in the background."""
    if not settings.routing.destructive_enabled:
        return "Re-indexing is disabled (ENABLE_DESTRUCTIVE_ACTIONS=0)."
    worker = threading.Thread(
        target=_run_ingestion_job, args=(r, t), daemon=True, name="rtfm-reingest"
    )
    worker.start()
    return (
        "Re-indexing started in the background - chewing through the docs "
        "corpus now. This takes a little while; ask me for my stats "
        "afterwards to see the new totals."
    )


def _warm_cache(r: Redis, t: TenantContext, session_id: str) -> str:
    """Pre-fill this tenant's semantic cache in the background (non-destructive)."""
    if not settings.warm.enabled:
        return "Cache warming is disabled (ENABLE_CACHE_WARM=0)."
    from rtfm_agent.routing.warm import start_background

    start_background(r, t)
    return (
        "Cache warming started in the background - I'm pre-answering your "
        "most-asked questions with the cheap model so future hits are "
        "instant. Ask me for my stats afterwards to see the new totals."
    )


def _run_ingestion_job(r: Redis, t: TenantContext) -> None:
    try:
        from rtfm_agent.ingestion.pipeline import run_ingestion

        summary = run_ingestion(r, t)
        logger.info("background reingest finished: %s", summary)
    except Exception as exc:
        logger.error("background reingest failed: %s", exc)
        publish(r, t, INGEST_FAILED,
                {"tenant": t.id, "docs_dir": docs_dir_for(t),
                 "error": str(exc), "background": True})
