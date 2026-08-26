"""Q&A endpoints: blocking POST /ask and the SSE stream GET /ask/stream."""

import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from rtfm_agent.api import state
from rtfm_agent.api.schemas import AskRequest, AskResponse, Citation
from rtfm_agent.common.metrics import (
    record_hit,
    record_miss,
    record_request,
    record_stale_answer,
)
from rtfm_agent.common.sessions import append as append_session
from rtfm_agent.common.tenancy import TenantContext, require_tenant
from rtfm_agent.config import settings
from rtfm_agent.ingestion import versioning
from rtfm_agent.llm import LLMError
from rtfm_agent.prompts import MEMORY_NONE_REPLY, REFUSAL
from rtfm_agent.retrieval import cache as cache_mod
from rtfm_agent.retrieval import scope as scope_mod
from rtfm_agent.retrieval.citations import dedupe_citations
from rtfm_agent.retrieval.rag import (
    answer_question,
    boost_chunks_by_memory,
    docs_dir_for,
    prepare_session_context,
    resolve_intent,
    stream_on_lane,
)
from rtfm_agent.retrieval.search import retrieve
from rtfm_agent.routing import actions as actions_mod
from rtfm_agent.routing import memory as memory_mod
from rtfm_agent.routing import warm as warm_mod
from rtfm_agent.embedder import embed_question

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, t: TenantContext = Depends(require_tenant)):
    session_id = req.session_id or uuid.uuid4().hex
    try:
        result = answer_question(state.get_redis(), req.question, session_id, t)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return AskResponse(
        session_id=session_id,
        tenant=t.id,
        **{**result, "citations": [Citation(**c) for c in result["citations"]]},
    )


def _event(name: str, data) -> str:
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {name}\ndata: {payload}\n\n"


@router.get("/ask/stream")
def ask_stream(question: str, session_id: str | None = None,
               t: TenantContext = Depends(require_tenant)):
    if len(question.strip()) < 3:
        raise HTTPException(status_code=422, detail="question too short")
    session_id = session_id or uuid.uuid4().hex
    r = state.get_redis()

    def generate():
        yield _event("session", {"session_id": session_id, "tenant": t.id})
        t0 = time.perf_counter()

        summary_note, hist_turns = prepare_session_context(r, t, session_id)

        # Semantic routing: one fused intent call decides the path.
        intent = resolve_intent(r, t, question, hist_turns)
        route = intent.route
        yield _event("route", {"route": route, "action": intent.action})

        if route == "chitchat":
            from rtfm_agent.prompts import build_chitchat_messages

            parts: list[str] = []
            try:
                for delta in stream_on_lane(
                    r, t, "economy",
                    build_chitchat_messages(question, hist_turns, summary_note),
                ):
                    parts.append(delta)
                    yield _event("token", delta)
            except LLMError as exc:
                record_request(r, t, errored=True)
                yield _event("error", str(exc))
                return
            answer = "".join(parts)
            record_request(r, t, completion_tokens=max(len(answer) // 4, 1))
            append_session(r, t, session_id, "user", question)
            append_session(r, t, session_id, "assistant", answer)
            memory_mod.save_turn(session_id, question, answer, t, r)
            yield _event("done", {"sources_consulted": 0, "cached": False,
                                  "documents_scoped": [], "route": route})
            return

        if route == "action":
            answer = actions_mod.dispatch(intent.action, r, t, session_id)
            record_request(r, t)
            append_session(r, t, session_id, "user", question)
            append_session(r, t, session_id, "assistant", answer)
            memory_mod.save_turn(session_id, question, answer, t, r)
            yield _event("token", answer)
            yield _event("done", {"sources_consulted": 0, "cached": False,
                                  "documents_scoped": [], "route": route,
                                  "action": intent.action})
            return

        if route == "memory":
            memories = memory_mod.search_memories(question, t)
            if memories:
                from rtfm_agent.prompts import build_memory_messages

                parts: list[str] = []
                try:
                    for delta in stream_on_lane(
                        r, t, "economy",
                        build_memory_messages(memories, question, hist_turns, summary_note),
                    ):
                        parts.append(delta)
                        yield _event("token", delta)
                except LLMError as exc:
                    record_request(r, t, errored=True)
                    yield _event("error", str(exc))
                    return
                answer = "".join(parts)
                record_request(r, t, completion_tokens=max(len(answer) // 4, 1))
            else:
                answer = MEMORY_NONE_REPLY
                record_request(r, t)
                yield _event("token", answer)
            append_session(r, t, session_id, "user", question)
            append_session(r, t, session_id, "assistant", answer)
            memory_mod.save_turn(session_id, question, answer, t, r)
            yield _event("done", {"sources_consulted": 0, "cached": False,
                                  "documents_scoped": [], "route": route})
            return

        # --------------------------------------------------------------
        # Doc route: hybrid search - follow-up rewrite + scope detection.
        # Popularity tracking feeds cache warming (no-op when disabled).
        # --------------------------------------------------------------
        warm_mod.track_question(r, t, question)
        rewritten_query = intent.query
        doc_filter = scope_mod.resolve_scope(r, t, question, intent.source_hint)
        qvec = embed_question(question)

        # A scoped question must never be served from the general cache.
        if doc_filter is None:
            hit = cache_mod.lookup(r, t, qvec)
            if hit is not None:
                staleness = versioning.answer_staleness(
                    r, t, docs_dir_for(t), cached_version=hit["corpus_version"]
                )
                if staleness["warning"]:
                    yield _event("warning", {
                        "stale": True, "message": staleness["warning"],
                    })
                yield _event("citations", hit["citations"])
                yield _event("token", hit["response"])
                yield _event("done", {"sources_consulted": len(hit["citations"]),
                                      "cached": True, "documents_scoped": [],
                                      "route": "doc", **staleness})
                record_hit(r, t, (time.perf_counter() - t0) * 1000)
                record_request(r, t)
                if staleness["stale"]:
                    record_stale_answer(r, t)
                append_session(r, t, session_id, "user", question)
                append_session(r, t, session_id, "assistant", hit["response"])
                memory_mod.save_turn(session_id, question, hit["response"], t, r)
                return

        if doc_filter is not None:
            chunks = retrieve(rewritten_query, r, t, qvec=qvec, doc_filter=doc_filter)
        elif rewritten_query != question:
            chunks = retrieve(rewritten_query, r, t, qvec=embed_question(rewritten_query))
        else:
            chunks = retrieve(question, r, t, qvec=qvec)

        memories = memory_mod.search_memories(question, t)
        chunks = boost_chunks_by_memory(chunks, memories)

        if not chunks and not memories and not hist_turns and not summary_note:
            record_miss(r, t, (time.perf_counter() - t0) * 1000)
            record_request(r, t)
            append_session(r, t, session_id, "user", question)
            append_session(r, t, session_id, "assistant", REFUSAL)
            yield _event("citations", [])
            yield _event("token", REFUSAL)
            yield _event("done", {"sources_consulted": 0, "cached": False,
                                  "documents_scoped": [], "route": "doc"})
            return
        citations = dedupe_citations(chunks) if chunks else []
        yield _event("citations", citations)
        yield _event("scope", sorted(doc_filter) if doc_filter else [])

        staleness = versioning.answer_staleness(r, t, docs_dir_for(t))
        if staleness["warning"]:
            yield _event("warning", {"stale": True, "message": staleness["warning"]})

        from rtfm_agent.prompts import build_doc_messages

        messages = build_doc_messages(question, chunks, hist_turns, memories, summary_note)
        parts: list[str] = []
        try:
            for delta in stream_on_lane(r, t, "generation", messages):
                parts.append(delta)
                yield _event("token", delta)
        except LLMError as exc:
            record_request(r, t, errored=True)
            yield _event("error", str(exc))
            return

        full_answer = "".join(parts)
        est_completion = max(len(full_answer) // 4, 1)
        record_miss(r, t, (time.perf_counter() - t0) * 1000)
        record_request(r, t, completion_tokens=est_completion)
        if staleness["stale"]:
            record_stale_answer(r, t)
        if chunks and doc_filter is None:
            cache_mod.store(
                r, t, question, qvec, full_answer,
                citations,
                corpus_version=versioning.current_version(r, t),
            )
        append_session(r, t, session_id, "user", question)
        append_session(r, t, session_id, "assistant", full_answer)
        memory_mod.save_turn(session_id, question, full_answer, t, r)
        yield _event("done", {"sources_consulted": len(chunks), "cached": False,
                              "documents_scoped": sorted(doc_filter) if doc_filter else [],
                              "route": "doc", **staleness})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
