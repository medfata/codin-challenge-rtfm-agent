"""Semantic-cache warming with the economy lane.

Redis doesn't care which model generated the content it stores - so popular
questions are answered offline by the cheap economy lane and written straight
into the semantic cache. Live users then hit Redis instead of burning the
capable generation pool; a live cache miss still generates with the capable
model as before.

Question popularity lives in a per-tenant ZSET `{prefix}qfreq`
(member=question text, score=ask count), updated on every doc-route question.
Warm runs are guarded by a per-tenant lock so jobs never overlap.

Fail-open everywhere: tracking or warming must never break asking - errors
log, publish an event when possible, and stop the run quietly.
"""

import logging
import threading

from redis import Redis

from rtfm_agent import llm as llm_client
from rtfm_agent.common.events import (
    CACHE_WARM_COMPLETED,
    CACHE_WARM_STARTED,
    publish,
)
from rtfm_agent.common.metrics import record_cache_warm, record_lane_call
from rtfm_agent.common.tenancy import TenantContext
from rtfm_agent.config import settings
from rtfm_agent.embedder import embed_question
from rtfm_agent.ingestion import versioning
from rtfm_agent.llm import LLMError
from rtfm_agent.prompts import SYSTEM_PROMPT, compose_user_message
from rtfm_agent.retrieval import cache as cache_mod
from rtfm_agent.retrieval import scope as scope_mod
from rtfm_agent.retrieval.citations import dedupe_citations
from rtfm_agent.retrieval.search import retrieve

logger = logging.getLogger(__name__)

LOCK_TTL_S = 300


def qfreq_key(t: TenantContext) -> str:
    return f"{t.prefix}qfreq"


def track_question(r: Redis, t: TenantContext, question: str) -> None:
    """Count one asked doc-route question for this tenant (bounded, TTL'd).

    Called on every doc-route request - including semantic-cache hits - so
    the popularity list reflects what users actually ask, not just misses.
    """
    if not settings.warm.enabled or not question.strip():
        return
    try:
        pipe = r.pipeline(transaction=False)
        pipe.zincrby(qfreq_key(t), 1, question[:2000])
        # Keep only the top-N entries; lowest scores fall off.
        pipe.zremrangebyrank(qfreq_key(t), 0, -(settings.warm.qfreq_max_entries + 1))
        pipe.expire(qfreq_key(t), settings.warm.qfreq_ttl_s)
        pipe.execute()
    except Exception as exc:
        logger.warning("question-frequency tracking failed (non-fatal): %s", exc)


def _warm_messages(question: str, chunks) -> list[dict]:
    """Standalone prompt: no session history, no memories (documented tradeoff)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": compose_user_message(question, chunks)},
    ]


def run_warm(r: Redis, t: TenantContext, top_n: int | None = None) -> dict:
    """Warm this tenant's cache for its most-asked questions.

    Per question: skip when a fresh cache hit already covers it, skip when
    document-scope detection fires (the live pipeline never lets scoped
    answers into the general cache - warmed entries must follow the same
    rule), skip when retrieval finds nothing (a refusal is never cached),
    otherwise generate with the economy lane and store the entry stamped
    with the current corpus version.

    Returns {warmed, skipped_hit, skipped_scoped, skipped_no_chunks, failed}.
    Raises RuntimeError when another warm job holds the lock.
    """
    summary = {
        "warmed": 0,
        "skipped_hit": 0,
        "skipped_scoped": 0,
        "skipped_no_chunks": 0,
        "failed": 0,
    }
    limit = top_n or settings.warm.top_n
    lock_key = f"{t.prefix}lock:warm"
    if not r.set(lock_key, "1", nx=True, ex=LOCK_TTL_S):
        raise RuntimeError("a cache warm run is already active for this tenant")
    try:
        questions_raw = r.zrevrange(qfreq_key(t), 0, limit - 1)
        questions = [
            q.decode(errors="replace") if isinstance(q, bytes) else str(q)
            for q in questions_raw
        ]
        logger.info("cache warm: %d candidate question(s) for tenant %s",
                    len(questions), t.id)
        for question in questions:
            # Large runs can outlive LOCK_TTL_S; touch the lock every step so
            # a slow generation never lets a second job slip in.
            try:
                r.expire(lock_key, LOCK_TTL_S)
            except Exception:
                pass
            try:
                if scope_mod.resolve_scope(r, t, question) is not None:
                    summary["skipped_scoped"] += 1
                    continue
                qvec = embed_question(question)
                if cache_mod.lookup(r, t, qvec) is not None:
                    summary["skipped_hit"] += 1
                    continue
                chunks = retrieve(question, r, t, qvec=qvec)
                if not chunks:
                    summary["skipped_no_chunks"] += 1
                    continue
                answer, _usage = llm_client.chat(
                    _warm_messages(question, chunks),
                    max_tokens=1024,
                    lane="economy",
                )
                record_lane_call(r, t, "economy")
                if answer.strip():
                    cache_mod.store(
                        r, t, question, qvec, answer,
                        dedupe_citations(chunks),
                        corpus_version=versioning.current_version(r, t),
                    )
                    summary["warmed"] += 1
                else:
                    summary["failed"] += 1
            except LLMError as exc:
                summary["failed"] += 1
                logger.warning("cache warm generation failed for %r: %s",
                               question[:80], exc)
            except Exception as exc:
                summary["failed"] += 1
                logger.warning("cache warm step failed for %r: %s",
                               question[:80], exc)
        logger.info("cache warm finished for %s: %s", t.id, summary)
        return summary
    finally:
        try:
            r.delete(lock_key)
        except Exception:
            pass


def start_background(r: Redis, t: TenantContext, top_n: int | None = None) -> None:
    """Run warm in a daemon thread; returns immediately (reingest pattern)."""
    worker = threading.Thread(
        target=_run_job, args=(r, t, top_n), daemon=True, name="rtfm-warm"
    )
    worker.start()


def _run_job(r: Redis, t: TenantContext, top_n: int | None) -> None:
    publish(r, t, CACHE_WARM_STARTED, {"tenant": t.id})
    try:
        summary = run_warm(r, t, top_n)
        record_cache_warm(r, t, summary["warmed"])
        publish(r, t, CACHE_WARM_COMPLETED, {"tenant": t.id, **summary})
    except RuntimeError as exc:
        logger.info("cache warm skipped (%s)", exc)
        publish(r, t, CACHE_WARM_COMPLETED, {"tenant": t.id, "error": str(exc)})
    except Exception as exc:
        logger.error("background cache warm failed: %s", exc)
        publish(r, t, CACHE_WARM_COMPLETED, {"tenant": t.id, "error": str(exc)})
