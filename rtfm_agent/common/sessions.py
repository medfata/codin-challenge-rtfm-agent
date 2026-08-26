"""Session memory: per-conversation history in Redis lists + rolling summaries.

Layout (all keys namespaced under the tenant prefix "t:{org}:"):
  LIST t:{org}:session:{id}:msgs     JSON entries {"role", "content", "ts"}
  STR  t:{org}:session:{id}:summary  rolling LLM summary of folded older turns
A sliding TTL refreshes on every write, so sessions expire after inactivity.
"""

import json
import logging
import time

from redis import Redis

from rtfm_agent.common.tenancy import TenantContext
from rtfm_agent.config import settings

logger = logging.getLogger(__name__)


def _key(t: TenantContext, session_id: str) -> str:
    return f"{t.prefix}session:{session_id}:msgs"


def _summary_key(t: TenantContext, session_id: str) -> str:
    return f"{t.prefix}session:{session_id}:summary"


def append(r: Redis, t: TenantContext, session_id: str, role: str, content: str) -> None:
    """Append one turn and refresh the session TTL."""
    entry = json.dumps({"role": role, "content": content, "ts": int(time.time())})
    key = _key(t, session_id)
    pipe = r.pipeline(transaction=False)
    pipe.rpush(key, entry)
    if settings.sessions.store_max > 0:
        pipe.ltrim(key, -settings.sessions.store_max, -1)
    pipe.expire(key, settings.sessions.ttl_seconds)
    pipe.expire(_summary_key(t, session_id), settings.sessions.ttl_seconds)
    pipe.execute()


def history(r: Redis, t: TenantContext, session_id: str,
            last_n: int | None = None) -> list[dict]:
    """Return the last N turns as [{role, content}] (content capped per msg)."""
    n = last_n or settings.sessions.history_max_messages
    raw = r.lrange(_key(t, session_id), -n, -1)
    out = []
    for item in raw:
        try:
            m = json.loads(item)
        except (json.JSONDecodeError, TypeError):
            continue
        content = m.get("content") or ""
        out.append({"role": m.get("role"), "content": content[:2000]})
    return out


def get_summary(r: Redis, t: TenantContext, session_id: str) -> str | None:
    v = r.get(_summary_key(t, session_id))
    return v.decode() if isinstance(v, bytes) else None


def set_summary(r: Redis, t: TenantContext, session_id: str, summary: str) -> None:
    pipe = r.pipeline(transaction=False)
    pipe.set(_summary_key(t, session_id), summary)
    pipe.expire(_summary_key(t, session_id), settings.sessions.ttl_seconds)
    pipe.execute()


def trim_to_recent(r: Redis, t: TenantContext, session_id: str) -> None:
    """Keep only the most recent verbatim window (older turns are summarised)."""
    r.ltrim(_key(t, session_id), -settings.sessions.recent_turns_verbatim, -1)


def build_prompt_context(r: Redis, t: TenantContext, session_id: str,
                         summarize_fn=None):
    """Split stored turns into (summary_note, prompt_turns).

    - Last recent_turns_verbatim messages always stay verbatim.
    - When older overflow exceeds summary_char_budget chars, fold it into the
      rolling summary via `summarize_fn` (LLM; fail-open keeps old behaviour).
    Returns (summary_text_or_None, turns_for_prompt).
    """
    all_msgs = history(r, t, session_id, last_n=settings.sessions.store_max)
    recent = all_msgs[-settings.sessions.recent_turns_verbatim:]
    older = all_msgs[:-settings.sessions.recent_turns_verbatim]
    summary = get_summary(r, t, session_id)

    if older:
        older_chars = sum(len(m["content"]) for m in older) + len(summary or "")
        if older_chars > settings.sessions.summary_char_budget and summarize_fn is not None:
            try:
                new_summary = summarize_fn(summary, older)
                if new_summary:
                    set_summary(r, t, session_id, new_summary)
                    trim_to_recent(r, t, session_id)
                    return new_summary, recent
            except Exception as exc:
                logger.warning("summarisation failed, keeping raw turns: %s", exc)

    combined = (f"{summary} " if summary else "") + " ".join(
        m["content"][:200] for m in older
    )
    note = summary or (combined.strip()[:600] or None)
    return note, older + recent


def touch(r: Redis, t: TenantContext, session_id: str) -> None:
    r.expire(_key(t, session_id), settings.sessions.ttl_seconds)


def ttl(r: Redis, t: TenantContext, session_id: str) -> int:
    """Seconds until the session expires (-2 if it doesn't exist)."""
    return r.ttl(_key(t, session_id))


def clear(r: Redis, t: TenantContext, session_id: str) -> int:
    removed = 0
    for key in (_key(t, session_id), _summary_key(t, session_id)):
        removed += r.delete(key)
    return removed
