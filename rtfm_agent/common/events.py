"""Real-time event notifications over a per-tenant Redis Stream.

Each tenant owns one stream, `t:{org}:events`. Producers XADD small JSON
envelopes (exact-MAXLEN ring buffer); the async SSE endpoint at
/events/stream XREADs them live and resumes from a client's Last-Event-ID
after reconnects. The stream entry id doubles as the SSE event id -
monotonic, so clients continue exactly where they left off (at-least-once
delivery; clients dedupe by id).

Two flavours, one semantics:
  * sync  (`publish`, `iter_events`)   - producers run in sync pipelines/CLI;
                                         `iter_events` remains for tests/tools.
  * async (`aiter_events`)             - the SSE subscriber loop; runs on the
                                         event loop so each client costs zero
                                         worker threads.

Fail-open everywhere: publishing or reading must never break ingestion,
asking, or memory saves. Errors are logged and swallowed.
"""

import json
import logging
import re
import time

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from rtfm_agent.common.tenancy import TenantContext
from rtfm_agent.config import settings

logger = logging.getLogger(__name__)

INGEST_STARTED = "ingest.started"
INGEST_COMPLETED = "ingest.completed"
INGEST_FAILED = "ingest.failed"
MEMORY_TURN_STORED = "memory.turn_stored"
CACHE_WARM_STARTED = "cache.warm_started"
CACHE_WARM_COMPLETED = "cache.warm_completed"
CRAWL_STAGED = "crawl.staged"
CRAWL_FAILED = "crawl.failed"

# Stream ids are "<milliseconds>-<sequence>"; anything else in a client's
# Last-Event-ID would poison every subsequent XREAD with errors.
_STREAM_ID_RE = re.compile(r"^\d{1,16}-\d{1,20}$")


def stream_key(t: TenantContext) -> str:
    return f"{t.prefix}events"


def normalize_last_id(raw) -> str:
    """Valid stream id, or "" when absent/malformed (resume falls back to $).

    Accepts the bytes ids redis-py returns as well as header strings.
    """
    if isinstance(raw, bytes):
        raw = raw.decode(errors="replace")
    raw = (raw or "").strip()
    return raw if _STREAM_ID_RE.match(raw) else ""


def publish(r: Redis, t: TenantContext, type_: str, data: dict) -> str | None:
    """XADD one event envelope + bump the counter in one pipeline.

    Returns the entry id (None when disabled/failed). Trimming is exact
    rather than approximate: approximate mode trims whole radix nodes
    (~100 entries), so short streams would never shrink.
    """
    if not settings.events.enabled:
        return None
    try:
        pipe = r.pipeline(transaction=False)
        pipe.xadd(
            stream_key(t),
            {
                "type": type_,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data": json.dumps(data, ensure_ascii=False, default=str),
            },
            maxlen=settings.events.stream_maxlen,
            approximate=False,
        )
        pipe.hincrby(t.metrics_key, "events_published_total", 1)
        entry_id = pipe.execute()[0]
    except Exception as exc:
        logger.warning("event publish failed (%s): %s", type_, exc)
        return None
    return entry_id


def _s(value) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _decode_entry(entry_id, fields) -> tuple[str, str, dict]:
    """(entry_id, type, data_dict); malformed entries degrade, never raise."""
    fmap = {_s(k): _s(v) for k, v in fields.items()}
    try:
        data = json.loads(fmap.get("data") or "{}")
        if not isinstance(data, dict):
            data = {"raw": data}
    except ValueError:
        data = {"raw": fmap.get("data")}
    return _s(entry_id), fmap.get("type") or "unknown", data


def resolve_start(client: Redis, t: TenantContext, last_event_id: str,
                  backlog: int) -> str:
    """Sync cursor for the live read loop (see aresolve_start for rules)."""
    last_event_id = normalize_last_id(last_event_id)
    if last_event_id:
        return last_event_id
    if backlog > 0:
        try:
            entries = client.xrevrange(stream_key(t), count=backlog + 1)
            if entries:
                if len(entries) > backlog:
                    return _s(entries[-1][0])
                return "0"
        except Exception as exc:
            logger.warning("event backlog probe failed: %s", exc)
    return "$"


async def aresolve_start(client: AsyncRedis, t: TenantContext,
                         last_event_id: str, backlog: int) -> str:
    """Cursor for the live read loop.

    XREAD returns entries with ids STRICTLY GREATER than the cursor, so:
    Last-Event-ID resumes by passing that id as-is (the client already has
    it); `backlog` N replays the most recent N entries by starting at their
    predecessor. Else "$" = only entries newer than now. A malformed
    Last-Event-ID is ignored instead of erroring forever.
    """
    last_event_id = normalize_last_id(last_event_id)
    if last_event_id:
        return last_event_id
    if backlog > 0:
        try:
            entries = await client.xrevrange(stream_key(t), count=backlog + 1)
            if entries:
                if len(entries) > backlog:
                    return _s(entries[-1][0])
                return "0"
        except Exception as exc:
            logger.warning("event backlog probe failed: %s", exc)
    return "$"


def iter_events(client: Redis, t: TenantContext, cursor: str):
    """Blocking sync XREAD loop; yields (entry_id, type, data) tuples.

    Yields None whenever a block window passes without new events so the
    caller can emit heartbeats. Connection hiccups also yield None instead
    of raising.
    """
    key = stream_key(t)
    block_ms = max(settings.events.heartbeat_s * 1000, 1000)
    while True:
        try:
            batches = client.xread({key: cursor}, block=block_ms, count=64)
        except Exception as exc:
            logger.warning("event xread failed: %s", exc)
            yield None
            continue
        if not batches:
            yield None
            continue
        for _key, entries in batches:
            for entry_id, fields in entries:
                parsed = _decode_entry(entry_id, fields)
                cursor = parsed[0]
                yield parsed


async def aiter_events(client: AsyncRedis, t: TenantContext, cursor: str):
    """Async XREAD loop for SSE subscribers - same contract as iter_events.

    Runs on the event loop (await releases between blocks), so subscribers
    cost no worker threads. Yields None on idle windows/errors for caller
    heartbeats. CancelledError propagates untouched for clean disconnects.
    """
    key = stream_key(t)
    block_ms = max(settings.events.heartbeat_s * 1000, 1000)
    while True:
        try:
            batches = await client.xread({key: cursor}, block=block_ms, count=64)
        except Exception as exc:
            logger.warning("event xread failed: %s", exc)
            yield None
            continue
        if not batches:
            yield None
            continue
        for _key, entries in batches:
            for entry_id, fields in entries:
                parsed = _decode_entry(entry_id, fields)
                cursor = parsed[0]
                yield parsed
