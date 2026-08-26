"""Shared runtime connections for the HTTP layer.

The FastAPI lifespan owns the sync + async Redis clients; route modules,
the RAG pipeline, and the MCP mount all obtain them from here instead of
reaching into module globals scattered across the codebase.
"""

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

_redis: Redis | None = None
_aredis: AsyncRedis | None = None


def set_runtime(r: Redis, ar: AsyncRedis) -> None:
    global _redis, _aredis
    _redis = r
    _aredis = ar


def get_redis() -> Redis:
    if _redis is None:
        raise RuntimeError("runtime not started: no Redis client (app lifespan)")
    return _redis


def get_async_redis() -> AsyncRedis:
    if _aredis is None:
        raise RuntimeError("runtime not started: no async Redis client (app lifespan)")
    return _aredis


async def aclose_runtime() -> None:
    global _aredis, _redis
    if _aredis is not None:
        await _aredis.aclose()
        _aredis = None
    if _redis is not None:
        _redis.close()
        _redis = None
