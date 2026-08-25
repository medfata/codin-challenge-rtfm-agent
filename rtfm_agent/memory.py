"""Long-term memory client for the Redis Agent Memory Server.

Every call fails open: if the memory server is unreachable we return empty
results and the assistant keeps working without personalization. All AMS
access is namespace-scoped by tenant so tenants never see each other's
memories; on any ambiguity we return empty rather than search unscoped.
"""

import logging

import httpx
from redis import Redis

from rtfm_agent import events as events_mod
from rtfm_agent.config import ENABLE_LONG_TERM_MEMORY, MEMORY_SEARCH_LIMIT, MEMORY_SERVER_URL
from rtfm_agent.tenancy import TenantContext

logger = logging.getLogger(__name__)
_warned = False

_timeout = httpx.Timeout(15)


def _warn_once(msg: str):
    global _warned
    if not _warned:
        logger.warning(msg)
        _warned = True


def is_healthy() -> bool:
    try:
        r = httpx.get(f"{MEMORY_SERVER_URL}/v1/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def search_memories(text: str, t: TenantContext, limit: int | None = None) -> list[dict]:
    """Tenant-scoped semantic search -> [{text, topics, entities}].

    Searches only the tenant's AMS namespace. A server that rejects the
    namespace filter (400/422) yields empty results, never an unfiltered search.
    """
    if not ENABLE_LONG_TERM_MEMORY or not text.strip():
        return []
    try:
        r = httpx.post(
            f"{MEMORY_SERVER_URL}/v1/long-term-memory/search",
            json={
                "text": text,
                "limit": limit or MEMORY_SEARCH_LIMIT,
                "filters": {"namespace": {"eq": t.id}},
            },
            timeout=_timeout,
        )
        if r.status_code in (400, 422):
            _warn_once("long-term memory search rejected namespace filter; returning empty for isolation")
            return []
        if r.status_code != 200:
            _warn_once(f"memory search HTTP {r.status_code}")
            return []
        out = []
        for m in r.json().get("memories", []):
            out.append({
                "text": m.get("text") or "",
                "topics": m.get("topics") or [],
                "entities": m.get("entities") or [],
            })
        return out
    except Exception as exc:
        _warn_once(f"memory search failed: {exc}")
        return []


def save_turn(session_id: str, question: str, answer: str, t: TenantContext,
              r: Redis | None = None) -> None:
    """Append a Q/A turn to the tenant's AMS working memory for this session.

    The record is written under the tenant's namespace so extraction only
    ever promotes facts within that tenant's scope. A `memory.turn_stored`
    event is published (when `r` is given) only after the server ACCEPTS
    the write (<400) - httpx does not raise on HTTP error statuses, so the
    response must be checked explicitly. Note this signals the saved turn,
    not long-term memories: AMS extracts those asynchronously inside its
    own worker and offers no hooks.
    """
    if not ENABLE_LONG_TERM_MEMORY:
        return
    url = f"{MEMORY_SERVER_URL}/v1/working-memory/{session_id}"
    try:
        messages: list[dict] = []
        existing = httpx.get(url, timeout=_timeout)
        if existing.status_code == 200:
            messages = [m for m in (existing.json().get("messages") or []) if isinstance(m, dict)]

        messages += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        # Bound the payload; older turns drop off.
        saved = httpx.put(url, json={"messages": messages[-20:], "namespace": t.id},
                          timeout=_timeout)
        if saved.status_code >= 400:
            _warn_once(f"working memory save rejected: HTTP {saved.status_code}")
        elif r is not None:
            events_mod.publish(r, t, events_mod.MEMORY_TURN_STORED,
                               {"session_id": session_id})
    except Exception as exc:
        _warn_once(f"working memory save failed: {exc}")
