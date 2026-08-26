"""Shared pytest fixtures.

Unit tests exercise pure logic with no I/O. Integration and API tests need
a reachable Redis (docker compose up redis) and are skipped otherwise.
"""

import os
import sys

import pytest

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rtfm_agent.common.tenancy import TenantContext  # noqa: E402

TEST_TENANT = "testrunner"


@pytest.fixture
def tenant() -> TenantContext:
    return TenantContext(TEST_TENANT)


def _redis_up() -> bool:
    try:
        import redis

        from rtfm_agent.config import settings

        r = redis.Redis.from_url(settings.redis.url, socket_connect_timeout=1)
        r.ping()
        return True
    except Exception:
        return False


REDIS_AVAILABLE = _redis_up()

requires_redis = pytest.mark.skipif(
    not REDIS_AVAILABLE, reason="live Redis required (docker compose up redis)"
)


@pytest.fixture
def clean_tenant_keys():
    """Delete every test-tenant key before AND after a test."""
    import redis

    from rtfm_agent.config import settings

    if not REDIS_AVAILABLE:
        yield None
        return
    r = redis.Redis.from_url(settings.redis.url, decode_responses=False)
    t = TenantContext(TEST_TENANT)

    def sweep():
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=f"{t.prefix}*", count=500)
            if keys:
                r.delete(*keys)
            if cursor == 0:
                break

    sweep()
    yield r
    sweep()


@pytest.fixture
def sample_docs_dir(tmp_path):
    """A tiny on-disk corpus for ingestion tests."""
    d = tmp_path / "docs"
    d.mkdir()
    (d / "intro.asc").write_text(
        "== Introduction\n\nGit is a distributed version control system.\n\n"
        "It snapshots your project. Every commit is a full snapshot.\n",
        encoding="utf-8",
    )
    (d / "advanced.asc").write_text(
        "=== Rebasing\n\nRebase replays your commits on top of another branch.\n\n"
        "This produces a linear history.\n",
        encoding="utf-8",
    )
    return d
