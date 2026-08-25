"""REST API: ingestion, RAG Q&A, semantic cache, session memory, long-term
memory, hybrid scoped search, conversation summarisation, observability,
semantic routing (doc/chitchat/action/memory) with conversational actions,
document versioning with stale-answer warnings, real-time event streaming
over per-tenant Redis Streams, and an MCP server mount."""

import contextlib
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from rtfm_agent import actions as actions_mod
from rtfm_agent import cache as cache_mod
from rtfm_agent import events as events_mod
from rtfm_agent import memory as memory_mod
from rtfm_agent import metrics as metrics_mod
from rtfm_agent import router as router_mod
from rtfm_agent import scope as scope_mod
from rtfm_agent import sessions as sessions_mod
from rtfm_agent import versions as versions_mod
from rtfm_agent.config import (
    DOCS_DIR,
    ENABLE_EVENTS,
    ENABLE_MCP,
    ENABLE_QUERY_REWRITE,
    ENABLE_ROUTING,
    EVENTS_CORS_ORIGINS,
    EVENTS_HEARTBEAT_S,
    REDIS_URL,
)
from rtfm_agent.llm import LLMError
from rtfm_agent import llm as llm_client
from rtfm_agent.prompts import (
    CHITCHAT_SYSTEM,
    MEMORY_NONE_REPLY,
    MEMORY_SYSTEM,
    REFUSAL,
    REWRITE_SYSTEM,
    SUMMARY_SYSTEM,
    SYSTEM_PROMPT,
    build_memory_prompt,
    build_rewrite_prompt,
    build_summary_prompt,
    compose_user_message,
)
from rtfm_agent.retrieval import get_embedder, retrieve
from rtfm_agent.tenancy import TenantContext, require_tenant, resolve_tenant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rtfm.api")

_r: Redis | None = None
_ar: AsyncRedis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _r, _ar
    _r = Redis.from_url(REDIS_URL, decode_responses=False)
    _r.ping()
    # Shared async pool for SSE subscribers: each /events/stream client
    # borrows a connection per XREAD await instead of owning one, so
    # hundreds of streams cost no extra sockets or worker threads.
    # socket_timeout MUST exceed the XREAD block window - redis-py 8.x async
    # otherwise falls back to the connect timeout for reads (5s), turning
    # every idle subscriber into a disconnect/reconnect churn loop.
    _ar = AsyncRedis.from_url(
        REDIS_URL, decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=EVENTS_HEARTBEAT_S + 5,
    )
    await _ar.ping()
    # Cache/doc indexes are per-tenant and self-heal lazily on first use.
    # The MCP mount's own lifespan never runs (Starlette drops sub-app
    # lifespans on Mount), so the session manager is entered here.
    try:
        async with contextlib.AsyncExitStack() as stack:
            if ENABLE_MCP and _mcp_mount is not None:
                await stack.enter_async_context(
                    _mcp_mount.lifespan_app.router.lifespan_context(_mcp_mount.lifespan_app)
                )
            yield
    finally:
        if _ar is not None:
            await _ar.aclose()
        _r.close()


app = FastAPI(title="RTFM For Me Agent", version="0.7.1", lifespan=lifespan)

# Cross-origin browser access for the SSE feed (and GETs generally). Scoped
# to read-only methods on purpose; tighten EVENTS_CORS_ORIGINS in production.
if EVENTS_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=EVENTS_CORS_ORIGINS,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["Last-Event-ID", "X-Tenant-Id"],
    )

# Built here but mounted at the END of this module (see bottom): the MCP app
# sits at root serving /mcp internally, and Starlette route order means it
# must be appended after every REST route.
_mcp_mount = None
if ENABLE_MCP:
    from rtfm_agent.mcp_server import create_mcp_server

    _mcp_mount = create_mcp_server(lambda: _r)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_docs_dir(t: TenantContext, override: str | None) -> Path:
    """Per-team corpus precedence: docs/<tenant>/ > validated override > DOCS_DIR."""
    team_dir = _PROJECT_ROOT / "docs" / t.id
    if team_dir.is_dir():
        return team_dir
    if override:
        resolved = Path(override).resolve()
        if not resolved.is_dir():
            raise HTTPException(
                status_code=422,
                detail=f"docs_dir '{override}' is not an existing directory",
            )
        try:
            resolved.relative_to(_PROJECT_ROOT)
        except ValueError:
            raise HTTPException(
                status_code=422, detail="docs_dir must stay inside the project root"
            )
        return resolved
    return Path(DOCS_DIR)


class IngestRequest(BaseModel):
    docs_dir: str | None = Field(default=None, max_length=500)


class IngestResponse(BaseModel):
    documents: int
    chunks_generated: int
    chunks_stored: int
    stale_keys_removed: int
    index_created: bool
    index: str
    embedding_dim: int
    duration_s: float
    tenant: str
    docs_dir: str
    corpus_version: int = 0
    digest: str = ""
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    session_id: str | None = Field(default=None, max_length=100)


class Citation(BaseModel):
    source_file: str
    section_heading: str
    chunk_pos: int
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources_consulted: int
    cached: bool = False
    session_id: str
    tenant: str = ""
    documents_scoped: list[str] = []
    route: str = "doc"
    action: str | None = None
    stale: bool = False
    warning: str | None = None


class MetricsResponse(BaseModel):
    requests_total: int
    errors_total: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_cached_ms: float
    avg_uncached_ms: float
    total_questions: int
    prompt_tokens_total: int
    completion_tokens_total: int
    estimated_cost_usd: float
    cache_size: int
    route_doc_total: int = 0
    route_chitchat_total: int = 0
    route_action_total: int = 0
    route_memory_total: int = 0
    stale_answers_served: int = 0
    mcp_calls_total: int = 0
    events_published_total: int = 0


def _embed_question(text: str) -> bytes:
    vec = get_embedder().embed([text])[0].astype(np.float32)
    return vec.tobytes()


def _docs_dir_for(t: TenantContext) -> str:
    """This tenant's default corpus directory, for version/drift checks."""
    return str(_resolve_docs_dir(t, None))


def _citations_from(chunks) -> list[Citation]:
    deduped: dict[tuple[str, str], Citation] = {}
    for c in chunks:
        key = (c["source_file"], c["section_heading"])
        if key not in deduped or c["score"] < deduped[key].score:
            deduped[key] = Citation(
                source_file=c["source_file"],
                section_heading=c["section_heading"],
                chunk_pos=c["chunk_pos"],
                score=round(c["score"], 4),
            )
    return sorted(deduped.values(), key=lambda x: x.score)


def _rewrite_query(question: str, hist_turns: list[dict]) -> tuple[str, str | None]:
    """Follow-up handling: standalone query + optional SOURCE document hint."""
    if not (hist_turns and ENABLE_QUERY_REWRITE):
        return question, None
    try:
        raw, _ = llm_client.chat(
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": build_rewrite_prompt(hist_turns, question)},
            ],
            temperature=0.0,
            max_tokens=512,
            model=llm_client.LLM_FAST_MODEL,
        )
        raw = raw.strip()
    except LLMError as exc:
        logger.warning("query rewrite failed (non-fatal): %s", exc)
        return question, None

    source_hint = None
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) >= 2 and lines[-1].upper().startswith("SOURCE:"):
        candidate = lines[-1][len("SOURCE:"):].strip().strip('"')
        if candidate:
            source_hint = candidate
        lines = lines[:-1]

    query = " ".join(lines).strip().strip('"')
    if (
        not query
        or len(query) < 8
        or query.lower().startswith(("user", "assistant"))
    ):
        return question, source_hint
    return query, source_hint


def _summarize_turns(existing_summary: str | None, older_msgs) -> str | None:
    """Fold older turns (+ previous summary) into a compact rolling summary."""
    raw, _ = llm_client.chat(
        [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": build_summary_prompt(existing_summary, older_msgs)},
        ],
        temperature=0.0,
        max_tokens=512,
        model=llm_client.LLM_FAST_MODEL,
    )
    return raw.strip() or None


_WORD_RE = re.compile(r"[a-z0-9]+")


def _boost_chunks_by_memory(chunks, memories):
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


def _build_messages(question: str, chunks, hist_turns: list[dict],
                    memories=None, summary_note: str | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if summary_note:
        messages.append({
            "role": "system",
            "content": f"Earlier in this conversation: {summary_note}",
        })
    messages.extend(hist_turns)
    messages.append({"role": "user", "content": compose_user_message(question, chunks, memories)})
    return messages


def _chitchat_messages(question: str, hist_turns: list[dict],
                       summary_note: str | None) -> list[dict]:
    messages = [{"role": "system", "content": CHITCHAT_SYSTEM}]
    if summary_note:
        messages.append({
            "role": "system",
            "content": f"Earlier in this conversation: {summary_note}",
        })
    messages.extend(hist_turns)
    messages.append({"role": "user", "content": question})
    return messages


def _memory_messages(memories, question: str, hist_turns: list[dict],
                     summary_note: str | None) -> list[dict]:
    messages = [{"role": "system", "content": MEMORY_SYSTEM}]
    if summary_note:
        messages.append({
            "role": "system",
            "content": f"Earlier in this conversation: {summary_note}",
        })
    messages.extend(hist_turns)
    messages.append({"role": "user", "content": build_memory_prompt(memories, question)})
    return messages


def _resolve_intent(question: str, hist_turns: list[dict], t: TenantContext):
    """Routing on: one fused LLM intent call; off: legacy rewrite behaviour."""
    if not ENABLE_ROUTING:
        rewritten, hint = _rewrite_query(question, hist_turns)
        return router_mod.RouteResult(query=rewritten, source_hint=hint)
    intent = router_mod.classify(question, hist_turns)
    metrics_mod.record_route(_r, t, intent.route)
    return intent


@app.get("/health")
def health():
    try:
        pong = _r.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis unreachable: {exc}")
    return {
        "status": "ok" if pong else "degraded",
        "redis": "up" if pong else "down",
        "llm_model": llm_client.LLM_MODEL,
        "llm_fast_model": llm_client.LLM_FAST_MODEL,
        "llm_configured": bool(llm_client.LLM_API_KEY),
        "fallback_llm_configured": bool(llm_client.FALLBACK_LLM_API_KEY),
        "cache_threshold": cache_mod.CACHE_THRESHOLD,
        "query_rewrite": ENABLE_QUERY_REWRITE,
        "routing": ENABLE_ROUTING,
        "session_ttl_s": sessions_mod.SESSION_TTL_SECONDS,
        "memory_server": "up" if memory_mod.is_healthy() else "down",
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest | None = None,
           t: TenantContext = Depends(require_tenant)):
    from rtfm_agent.ingest import run_ingestion

    path = _resolve_docs_dir(t, req.docs_dir if req else None)
    try:
        summary = run_ingestion(_r, t, docs_dir=str(path))
    except Exception as exc:
        events_mod.publish(_r, t, events_mod.INGEST_FAILED,
                           {"tenant": t.id, "docs_dir": str(path), "error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    return IngestResponse(docs_dir=str(path), **summary)


@app.get("/docs/status")
def docs_status(t: TenantContext = Depends(require_tenant)):
    """Corpus version plus a forced on-disk drift scan for this tenant."""
    return versions_mod.status_report(_r, t, _docs_dir_for(t))


def _pipeline(question: str, session_id: str, t: TenantContext) -> dict:
    """Shared RAG pipeline: summarise -> scope -> cache -> recall -> answer.

    Returns {answer, citations, sources_consulted, cached, documents_scoped}.
    Raises HTTPException(502) on hard LLM failure.
    """
    t0 = time.perf_counter()

    # Session context: verbatim recent turns + rolling summary.
    summary_note, hist_turns = sessions_mod.build_prompt_context(
        _r, t, session_id, summarize_fn=_summarize_turns
    )

    # Semantic routing: one fused intent call decides the path.
    intent = _resolve_intent(question, hist_turns, t)
    route = intent.route

    # Chitchat: persona reply, no cache/retrieval/memory search.
    if route == "chitchat":
        try:
            answer, usage = llm_client.chat(
                _chitchat_messages(question, hist_turns, summary_note)
            )
        except LLMError as exc:
            metrics_mod.record_request(_r, t, errored=True)
            raise HTTPException(status_code=502, detail=str(exc))
        metrics_mod.record_request(
            _r, t,
            prompt_tokens=(usage or {}).get("prompt_tokens", 0) or 0,
            completion_tokens=(usage or {}).get("completion_tokens", 0) or 0,
        )
        sessions_mod.append(_r, t, session_id, "user", question)
        sessions_mod.append(_r, t, session_id, "assistant", answer)
        memory_mod.save_turn(session_id, question, answer, t, _r)
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
        answer = actions_mod.dispatch(intent.action, _r, t, session_id)
        metrics_mod.record_request(_r, t)
        sessions_mod.append(_r, t, session_id, "user", question)
        sessions_mod.append(_r, t, session_id, "assistant", answer)
        memory_mod.save_turn(session_id, question, answer, t, _r)
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
            try:
                answer, usage = llm_client.chat(
                    _memory_messages(memories, question, hist_turns, summary_note)
                )
            except LLMError as exc:
                metrics_mod.record_request(_r, t, errored=True)
                raise HTTPException(status_code=502, detail=str(exc))
        else:
            answer, usage = MEMORY_NONE_REPLY, None
        metrics_mod.record_request(
            _r, t,
            prompt_tokens=(usage or {}).get("prompt_tokens", 0) or 0,
            completion_tokens=(usage or {}).get("completion_tokens", 0) or 0,
        )
        sessions_mod.append(_r, t, session_id, "user", question)
        sessions_mod.append(_r, t, session_id, "assistant", answer)
        memory_mod.save_turn(session_id, question, answer, t, _r)
        return {
            "answer": answer,
            "citations": [],
            "sources_consulted": 0,
            "cached": False,
            "documents_scoped": [],
            "route": route,
        }

    # Doc route: hybrid search - follow-up rewrite + document scope detection.
    rewritten_query = intent.query
    doc_filter = scope_mod.resolve_scope(_r, t, question, intent.source_hint)
    qvec = _embed_question(question)

    # A scoped question must never be served from (or pollute) the general cache.
    if doc_filter is None:
        hit = cache_mod.lookup(_r, t, qvec)
        if hit is not None:
            staleness = versions_mod.answer_staleness(
                _r, t, _docs_dir_for(t), cached_version=hit["corpus_version"]
            )
            if staleness["stale"]:
                metrics_mod.record_stale_answer(_r, t)
            latency = (time.perf_counter() - t0) * 1000
            metrics_mod.record_hit(_r, t, latency)
            metrics_mod.record_request(_r, t)
            sessions_mod.append(_r, t, session_id, "user", question)
            sessions_mod.append(_r, t, session_id, "assistant", hit["response"])
            memory_mod.save_turn(session_id, question, hit["response"], t, _r)
            return {
                "answer": hit["response"],
                "citations": [Citation(**c) for c in hit["citations"]],
                "sources_consulted": len(hit["citations"]),
                "cached": True,
                "documents_scoped": [],
                "route": "doc",
                **staleness,
            }

    if doc_filter is not None:
        chunks = retrieve(rewritten_query, _r, t, qvec=qvec, doc_filter=doc_filter)
    elif rewritten_query != question:
        chunks = retrieve(rewritten_query, _r, t, qvec=_embed_question(rewritten_query))
    else:
        chunks = retrieve(question, _r, t, qvec=qvec)

    # Long-term memory: durable facts about this user.
    memories = memory_mod.search_memories(question, t)
    chunks = _boost_chunks_by_memory(chunks, memories)

    if not chunks and not memories and not hist_turns and not summary_note:
        latency = (time.perf_counter() - t0) * 1000
        metrics_mod.record_miss(_r, t, latency)
        metrics_mod.record_request(_r, t)
        sessions_mod.append(_r, t, session_id, "user", question)
        sessions_mod.append(_r, t, session_id, "assistant", REFUSAL)
        return {
            "answer": REFUSAL,
            "citations": [],
            "sources_consulted": 0,
            "cached": False,
            "documents_scoped": [],
            "route": "doc",
        }

    citations = _citations_from(chunks) if chunks else []
    messages = _build_messages(question, chunks, hist_turns, memories, summary_note)
    try:
        answer, usage = llm_client.chat(messages)
    except LLMError as exc:
        latency = (time.perf_counter() - t0) * 1000
        metrics_mod.record_request(_r, t, errored=True)
        raise HTTPException(status_code=502, detail=str(exc))

    latency = (time.perf_counter() - t0) * 1000
    metrics_mod.record_miss(_r, t, latency)
    metrics_mod.record_request(
        _r, t,
        prompt_tokens=(usage or {}).get("prompt_tokens", 0) or 0,
        completion_tokens=(usage or {}).get("completion_tokens", 0) or 0,
    )
    staleness = versions_mod.answer_staleness(_r, t, _docs_dir_for(t))
    if staleness["stale"]:
        metrics_mod.record_stale_answer(_r, t)
    if chunks and doc_filter is None:
        cache_mod.store(
            _r, t, question, qvec, answer,
            [c.model_dump() for c in citations],
            corpus_version=versions_mod.current_version(_r, t),
        )
    sessions_mod.append(_r, t, session_id, "user", question)
    sessions_mod.append(_r, t, session_id, "assistant", answer)
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


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, t: TenantContext = Depends(require_tenant)):
    session_id = req.session_id or uuid.uuid4().hex
    result = _pipeline(req.question, session_id, t)
    return AskResponse(session_id=session_id, tenant=t.id, **result)


@app.get("/ask/stream")
def ask_stream(question: str, session_id: str | None = None,
               t: TenantContext = Depends(require_tenant)):
    if len(question.strip()) < 3:
        raise HTTPException(status_code=422, detail="question too short")
    session_id = session_id or uuid.uuid4().hex

    def event(name: str, data) -> str:
        payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
        return f"event: {name}\ndata: {payload}\n\n"

    def generate():
        yield event("session", {"session_id": session_id, "tenant": t.id})
        t0 = time.perf_counter()

        summary_note, hist_turns = sessions_mod.build_prompt_context(
            _r, t, session_id, summarize_fn=_summarize_turns
        )

        # Semantic routing: one fused intent call decides the path.
        intent = _resolve_intent(question, hist_turns, t)
        route = intent.route
        yield event("route", {"route": route, "action": intent.action})

        if route == "chitchat":
            parts: list[str] = []
            try:
                for delta in llm_client.stream_chat(
                    _chitchat_messages(question, hist_turns, summary_note)
                ):
                    parts.append(delta)
                    yield event("token", delta)
            except LLMError as exc:
                metrics_mod.record_request(_r, t, errored=True)
                yield event("error", str(exc))
                return
            answer = "".join(parts)
            metrics_mod.record_request(_r, t, completion_tokens=max(len(answer) // 4, 1))
            sessions_mod.append(_r, t, session_id, "user", question)
            sessions_mod.append(_r, t, session_id, "assistant", answer)
            memory_mod.save_turn(session_id, question, answer, t, _r)
            yield event("done", {"sources_consulted": 0, "cached": False,
                                 "documents_scoped": [], "route": route})
            return

        if route == "action":
            answer = actions_mod.dispatch(intent.action, _r, t, session_id)
            metrics_mod.record_request(_r, t)
            sessions_mod.append(_r, t, session_id, "user", question)
            sessions_mod.append(_r, t, session_id, "assistant", answer)
            memory_mod.save_turn(session_id, question, answer, t, _r)
            yield event("token", answer)
            yield event("done", {"sources_consulted": 0, "cached": False,
                                 "documents_scoped": [], "route": route,
                                 "action": intent.action})
            return

        if route == "memory":
            memories = memory_mod.search_memories(question, t)
            if memories:
                parts: list[str] = []
                try:
                    for delta in llm_client.stream_chat(
                        _memory_messages(memories, question, hist_turns, summary_note)
                    ):
                        parts.append(delta)
                        yield event("token", delta)
                except LLMError as exc:
                    metrics_mod.record_request(_r, t, errored=True)
                    yield event("error", str(exc))
                    return
                answer = "".join(parts)
                metrics_mod.record_request(_r, t, completion_tokens=max(len(answer) // 4, 1))
            else:
                answer = MEMORY_NONE_REPLY
                metrics_mod.record_request(_r, t)
                yield event("token", answer)
            sessions_mod.append(_r, t, session_id, "user", question)
            sessions_mod.append(_r, t, session_id, "assistant", answer)
            memory_mod.save_turn(session_id, question, answer, t, _r)
            yield event("done", {"sources_consulted": 0, "cached": False,
                                 "documents_scoped": [], "route": route})
            return

        # Doc route: hybrid search - follow-up rewrite + document scope detection.
        rewritten_query = intent.query
        doc_filter = scope_mod.resolve_scope(_r, t, question, intent.source_hint)
        qvec = _embed_question(question)

        if doc_filter is None:
            hit = cache_mod.lookup(_r, t, qvec)
            if hit is not None:
                staleness = versions_mod.answer_staleness(
                    _r, t, _docs_dir_for(t), cached_version=hit["corpus_version"]
                )
                if staleness["warning"]:
                    yield event("warning", {
                        "stale": True, "message": staleness["warning"],
                    })
                yield event("citations", hit["citations"])
                yield event("token", hit["response"])
                yield event("done", {"sources_consulted": len(hit["citations"]),
                                     "cached": True, "documents_scoped": [],
                                     "route": "doc", **staleness})
                metrics_mod.record_hit(_r, t, (time.perf_counter() - t0) * 1000)
                metrics_mod.record_request(_r, t)
                if staleness["stale"]:
                    metrics_mod.record_stale_answer(_r, t)
                sessions_mod.append(_r, t, session_id, "user", question)
                sessions_mod.append(_r, t, session_id, "assistant", hit["response"])
                memory_mod.save_turn(session_id, question, hit["response"], t, _r)
                return

        if doc_filter is not None:
            chunks = retrieve(rewritten_query, _r, t, qvec=qvec, doc_filter=doc_filter)
        elif rewritten_query != question:
            chunks = retrieve(rewritten_query, _r, t, qvec=_embed_question(rewritten_query))
        else:
            chunks = retrieve(question, _r, t, qvec=qvec)

        memories = memory_mod.search_memories(question, t)
        chunks = _boost_chunks_by_memory(chunks, memories)

        if not chunks and not memories and not hist_turns and not summary_note:
            metrics_mod.record_miss(_r, t, (time.perf_counter() - t0) * 1000)
            metrics_mod.record_request(_r, t)
            sessions_mod.append(_r, t, session_id, "user", question)
            sessions_mod.append(_r, t, session_id, "assistant", REFUSAL)
            yield event("citations", [])
            yield event("token", REFUSAL)
            yield event("done", {"sources_consulted": 0, "cached": False,
                                 "documents_scoped": [], "route": "doc"})
            return
        citations = _citations_from(chunks) if chunks else []
        yield event("citations", [c.model_dump() for c in citations])
        yield event("scope", sorted(doc_filter) if doc_filter else [])

        staleness = versions_mod.answer_staleness(_r, t, _docs_dir_for(t))
        if staleness["warning"]:
            yield event("warning", {"stale": True, "message": staleness["warning"]})

        messages = _build_messages(question, chunks, hist_turns, memories, summary_note)
        parts: list[str] = []
        usage: dict = {}
        try:
            for delta in llm_client.stream_chat(messages):
                parts.append(delta)
                yield event("token", delta)
        except LLMError as exc:
            metrics_mod.record_request(_r, t, errored=True)
            yield event("error", str(exc))
            return

        full_answer = "".join(parts)
        est_completion = max(len(full_answer) // 4, 1)
        metrics_mod.record_miss(_r, t, (time.perf_counter() - t0) * 1000)
        metrics_mod.record_request(_r, t, completion_tokens=est_completion)
        if staleness["stale"]:
            metrics_mod.record_stale_answer(_r, t)
        if chunks and doc_filter is None:
            cache_mod.store(
                _r, t, question, qvec, full_answer,
                [c.model_dump() for c in citations],
                corpus_version=versions_mod.current_version(_r, t),
            )
        sessions_mod.append(_r, t, session_id, "user", question)
        sessions_mod.append(_r, t, session_id, "assistant", full_answer)
        memory_mod.save_turn(session_id, question, full_answer, t, _r)
        yield event("done", {"sources_consulted": len(chunks), "cached": False,
                             "documents_scoped": sorted(doc_filter) if doc_filter else [],
                             "route": "doc", **staleness})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _event_tenant(x_tenant_id: str = Header(default=""),
                  tenant: str | None = Query(default=None)) -> TenantContext:
    """Tenant resolution for /events/stream.

    Browsers' native EventSource cannot set custom headers, so ?tenant= is
    accepted as an equivalent trusted identity (same validation, same trust
    level as the X-Tenant-Id header). Sending both with different values is
    rejected rather than silently picking one.
    """
    if tenant and x_tenant_id.strip() and tenant.strip().lower() != x_tenant_id.strip().lower():
        raise HTTPException(
            status_code=422,
            detail="conflicting tenant identity: ?tenant= and X-Tenant-Id disagree",
        )
    return resolve_tenant(tenant or x_tenant_id)


@app.get("/events/stream")
async def events_stream(
    backlog: int = Query(default=0, ge=0, le=100),
    tenant: str | None = Query(default=None),
    last_event_id: str = Header(default=""),
    t: TenantContext = Depends(_event_tenant),
):
    """SSE feed of this tenant's real-time events.

    Emits ingest.started/completed/failed and memory.turn_stored frames.
    Reconnects resume from the browser's Last-Event-ID header (stream entry
    ids double as SSE ids; malformed ids are ignored, falling back to live
    tail); ?backlog=N replays the N most recent entries on a fresh connect.
    Heartbeat comments keep proxies from idling out.

    The read loop is fully async on the shared Redis pool - subscribers
    cost no worker threads, so the sync endpoint pool stays free.
    """
    if not ENABLE_EVENTS:
        raise HTTPException(
            status_code=503, detail="event streaming is disabled (ENABLE_EVENTS=0)"
        )

    # Probe eagerly (shared async pool) so a dead Redis is a clean 503
    # instead of a stream that only ever emits heartbeats.
    try:
        await _ar.ping()
        cursor = await events_mod.aresolve_start(
            _ar, t, events_mod.normalize_last_id(last_event_id), backlog
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis unreachable: {exc}")

    async def generate():
        yield "retry: 3000\n\n"
        yield f": connected tenant={t.id}\n\n"
        async for item in events_mod.aiter_events(_ar, t, cursor):
            if item is None:
                yield ": ping\n\n"
                continue
            entry_id, etype, data = item
            payload = json.dumps(data, ensure_ascii=False)
            yield f"id: {entry_id}\nevent: {etype}\ndata: {payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/demo/events", include_in_schema=False)
def events_demo_page():
    """Serve the event-feed demo page same-origin (no CORS setup needed)."""
    page = _PROJECT_ROOT / "scripts" / "events_demo.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="demo page not found")
    return FileResponse(page, media_type="text/html")


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics(t: TenantContext = Depends(require_tenant)):
    snap = metrics_mod.snapshot(_r, t)
    snap["cache_size"] = cache_mod.count_entries(_r, t)
    return MetricsResponse(**snap)


@app.post("/cache/flush")
def flush_cache(t: TenantContext = Depends(require_tenant)):
    removed = cache_mod.flush(_r, t)
    return {"removed": removed, "tenant": t.id}


@app.get("/sessions/{session_id}/history")
def get_session_history(session_id: str, limit: int = 50,
                        t: TenantContext = Depends(require_tenant)):
    msgs = sessions_mod.history(_r, t, session_id, last_n=limit)
    return {
        "session_id": session_id,
        "ttl_seconds": sessions_mod.ttl(_r, t, session_id),
        "message_count": len(msgs),
        "summary": sessions_mod.get_summary(_r, t, session_id),
        "messages": msgs,
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, t: TenantContext = Depends(require_tenant)):
    removed = sessions_mod.clear(_r, t, session_id)
    if not removed:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "deleted": True}


# Must come after every REST route: a root Mount swallows all later routes.
if ENABLE_MCP and _mcp_mount is not None:
    app.mount("/", _mcp_mount.app)

