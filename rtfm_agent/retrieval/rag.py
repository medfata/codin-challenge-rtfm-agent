"""RAG pipeline orchestration: summarise -> route -> cache -> recall -> answer.

`answer_question` is the canonical doc-route pipeline shared by the REST
endpoint and the MCP tool server; the SSE stream composes the same
primitives (resolve_intent, lane helpers, cache/search/scope) into an
event generator. Citations are plain dicts so every transport can shape
them freely.
"""

import logging
import re
import time

from redis import Redis

from rtfm_agent import llm as llm_client
from rtfm_agent.common.metrics import (
    record_hit,
    record_lane_call,
    record_miss,
    record_request,
    record_route,
    record_stale_answer,
)
from rtfm_agent.common.sessions import append as append_session
from rtfm_agent.common.sessions import build_prompt_context
from rtfm_agent.common.tenancy import TenantContext
from rtfm_agent.config import settings
from rtfm_agent.embedder import embed_question
from rtfm_agent.ingestion import versioning
from rtfm_agent.llm import LLMError
from rtfm_agent.prompts import MEMORY_NONE_REPLY, REFUSAL, build_doc_messages
from rtfm_agent.retrieval import cache as cache_mod
from rtfm_agent.retrieval import scope as scope_mod
from rtfm_agent.retrieval.citations import dedupe_citations
from rtfm_agent.retrieval.rewrite import rewrite_query, summarize_turns
from rtfm_agent.retrieval.search import retrieve
from rtfm_agent.routing import actions as actions_mod
from rtfm_agent.routing import memory as memory_mod
from rtfm_agent.routing import warm as warm_mod
from rtfm_agent.routing.intent import RouteResult, classify

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# LLM lane helpers (call + success counter)
# ---------------------------------------------------------------------------


def chat_on_lane(r: Redis, t: TenantContext, lane: str, messages: list[dict], **kwargs):
    """Blocking chat on a named lane + success counter."""
    out = llm_client.chat(messages, lane=lane, **kwargs)
    record_lane_call(r, t, lane)
    return out


def stream_on_lane(r: Redis, t: TenantContext, lane: str, messages: list[dict], **kwargs):
    """Streaming chat on a named lane; counted once the stream completes."""
    for delta in llm_client.stream_chat(messages, lane=lane, **kwargs):
        yield delta
    record_lane_call(r, t, lane)


# ---------------------------------------------------------------------------
# Intent resolution (fused LLM call; off -> legacy rewrite behaviour)
# ---------------------------------------------------------------------------


def resolve_intent(r: Redis, t: TenantContext, question: str,
                   hist_turns: list[dict]) -> RouteResult:
    """One fused LLM intent call decides the path before any RAG work."""
    if not settings.routing.enabled:
        rewritten, hint = rewrite_query(question, hist_turns, r=r, t=t)
        return RouteResult(query=rewritten, source_hint=hint)
    intent = classify(question, hist_turns)
    # The intent call rides the primary endpoint on the fast model.
    record_lane_call(r, t, "fast")
    record_route(r, t, intent.route)
    return intent


def prepare_session_context(r: Redis, t: TenantContext, session_id: str):
    """Verbatim recent turns + rolling summary for one request."""
    return build_prompt_context(
        r, t, session_id,
        summarize_fn=lambda existing, older: summarize_turns(existing, older, r=r, t=t),
    )


# ---------------------------------------------------------------------------
# Chunk boosting from long-term memories
# ---------------------------------------------------------------------------


def boost_chunks_by_memory(chunks, memories):
    """Soft-prioritise chunks whose metadata matches remembered topics."""
    if not chunks or not memories:
        return chunks
    words: set[str] = set()
    for m in memories:
        for tok in list(m.get("topics") or []) + list(m.get("entities") or []):
            words.update(_WORD_RE.findall(str(tok).lower()))
    if not words:
        return chunks

    def score(c):
        meta = f"{c['source_file']} {c['section_heading']}".lower()
        return len(set(_WORD_RE.findall(meta)) & words)

    return sorted(chunks, key=score, reverse=True)


def docs_dir_for(t: TenantContext) -> str:
    """This tenant's default corpus directory (version/drift checks)."""
    from rtfm_agent.common.paths import docs_dir_for as _docs_dir_for

    return _docs_dir_for(t)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def answer_question(r: Redis, question: str, session_id: str,
                    t: TenantContext) -> dict:
    """Shared RAG pipeline for one message.

    Returns {answer, citations, sources_consulted, cached, documents_scoped,
             route[, action][, stale][, warning]}.
    Raises HTTPException-mapped LLMError(502 upstream) on hard failure -
    callers translate LLMError to their transport's error shape.
    """
    t0 = time.perf_counter()

    # Session context: verbatim recent turns + rolling summary.
    summary_note, hist_turns = prepare_session_context(r, t, session_id)

    # Semantic routing: one fused intent call decides the path.
    intent = resolve_intent(r, t, question, hist_turns)
    route = intent.route

    # Chitchat: persona reply, no cache/retrieval/memory search.
    if route == "chitchat":
        from rtfm_agent.prompts import build_chitchat_messages

        try:
            answer, usage = chat_on_lane(
                r, t, "economy",
                build_chitchat_messages(question, hist_turns, summary_note),
            )
        except LLMError:
            record_request(r, t, errored=True)
            raise
        record_request(
            r, t,
            prompt_tokens=(usage or {}).get("prompt_tokens", 0) or 0,
            completion_tokens=(usage or {}).get("completion_tokens", 0) or 0,
        )
        append_session(r, t, session_id, "user", question)
        append_session(r, t, session_id, "assistant", answer)
        memory_mod.save_turn(session_id, question, answer, t, r)
        return {
            "answer": answer,
            "citations": [],
            "sources_consulted": 0,
            "cached": False,
            "documents_scoped": [],
            "route": route,
        }

    # Action: deterministic handler reply, no generation call.
    if route == "action":
        answer = actions_mod.dispatch(intent.action, r, t, session_id)
        record_request(r, t)
        append_session(r, t, session_id, "user", question)
        append_session(r, t, session_id, "assistant", answer)
        memory_mod.save_turn(session_id, question, answer, t, r)
        return {
            "answer": answer,
            "citations": [],
            "sources_consulted": 0,
            "cached": False,
            "documents_scoped": [],
            "route": route,
            "action": intent.action,
        }

    # Memory: recall durable facts about this user and synthesise.
    if route == "memory":
        memories = memory_mod.search_memories(question, t)
        if memories:
            from rtfm_agent.prompts import build_memory_messages

            try:
                answer, usage = chat_on_lane(
                    r, t, "economy",
                    build_memory_messages(memories, question, hist_turns, summary_note),
                )
            except LLMError:
                record_request(r, t, errored=True)
                raise
        else:
            answer, usage = MEMORY_NONE_REPLY, None
        record_request(
            r, t,
            prompt_tokens=(usage or {}).get("prompt_tokens", 0) or 0,
            completion_tokens=(usage or {}).get("completion_tokens", 0) or 0,
        )
        append_session(r, t, session_id, "user", question)
        append_session(r, t, session_id, "assistant", answer)
        memory_mod.save_turn(session_id, question, answer, t, r)
        return {
            "answer": answer,
            "citations": [],
            "sources_consulted": 0,
            "cached": False,
            "documents_scoped": [],
            "route": route,
        }

    # ------------------------------------------------------------------
    # Doc route: hybrid search - follow-up rewrite + document scope detection.
    # Popularity tracking feeds cache warming (no-op when warming disabled).
    # ------------------------------------------------------------------
    warm_mod.track_question(r, t, question)
    rewritten_query = intent.query
    doc_filter = scope_mod.resolve_scope(r, t, question, intent.source_hint)
    qvec = embed_question(question)

    # A scoped question must never be served from (or pollute) the general cache.
    if doc_filter is None:
        hit = cache_mod.lookup(r, t, qvec)
        if hit is not None:
            staleness = versioning.answer_staleness(
                r, t, docs_dir_for(t), cached_version=hit["corpus_version"]
            )
            if staleness["stale"]:
                record_stale_answer(r, t)
            latency = (time.perf_counter() - t0) * 1000
            record_hit(r, t, latency)
            record_request(r, t)
            append_session(r, t, session_id, "user", question)
            append_session(r, t, session_id, "assistant", hit["response"])
            memory_mod.save_turn(session_id, question, hit["response"], t, r)
            return {
                "answer": hit["response"],
                "citations": [dict(c) for c in hit["citations"]],
                "sources_consulted": len(hit["citations"]),
                "cached": True,
                "documents_scoped": [],
                "route": "doc",
                **staleness,
            }

    if doc_filter is not None:
        chunks = retrieve(rewritten_query, r, t, qvec=qvec, doc_filter=doc_filter)
    elif rewritten_query != question:
        chunks = retrieve(rewritten_query, r, t, qvec=embed_question(rewritten_query))
    else:
        chunks = retrieve(question, r, t, qvec=qvec)

    # Long-term memory: durable facts about this user.
    memories = memory_mod.search_memories(question, t)
    chunks = boost_chunks_by_memory(chunks, memories)

    if not chunks and not memories and not hist_turns and not summary_note:
        latency = (time.perf_counter() - t0) * 1000
        record_miss(r, t, latency)
        record_request(r, t)
        append_session(r, t, session_id, "user", question)
        append_session(r, t, session_id, "assistant", REFUSAL)
        return {
            "answer": REFUSAL,
            "citations": [],
            "sources_consulted": 0,
            "cached": False,
            "documents_scoped": [],
            "route": "doc",
        }

    citations = dedupe_citations(chunks) if chunks else []
    messages = build_doc_messages(question, chunks, hist_turns, memories, summary_note)
    try:
        answer, usage = chat_on_lane(r, t, "generation", messages)
    except LLMError:
        latency = (time.perf_counter() - t0) * 1000
        record_request(r, t, errored=True)
        raise

    latency = (time.perf_counter() - t0) * 1000
    record_miss(r, t, latency)
    record_request(
        r, t,
        prompt_tokens=(usage or {}).get("prompt_tokens", 0) or 0,
        completion_tokens=(usage or {}).get("completion_tokens", 0) or 0,
    )
    staleness = versioning.answer_staleness(r, t, docs_dir_for(t))
    if staleness["stale"]:
        record_stale_answer(r, t)
    if chunks and doc_filter is None:
        cache_mod.store(
            r, t, question, qvec, answer,
            citations,
            corpus_version=versioning.current_version(r, t),
        )
    append_session(r, t, session_id, "user", question)
    append_session(r, t, session_id, "assistant", answer)
    memory_mod.save_turn(session_id, question, answer, t)
    return {
        "answer": answer,
        "citations": citations,
        "sources_consulted": len(chunks),
        "cached": False,
        "documents_scoped": sorted(doc_filter) if doc_filter else [],
        "route": "doc",
        **staleness,
    }
