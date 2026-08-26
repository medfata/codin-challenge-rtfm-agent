"""Integration tests: semantic cache + sessions against live Redis."""

from rtfm_agent.common.sessions import append, clear, get_summary, history
from rtfm_agent.common.tenancy import TenantContext
from rtfm_agent.retrieval.cache import count_entries, flush, store
from tests.conftest import TEST_TENANT, requires_redis


def _tenant():
    return TenantContext(TEST_TENANT)


@requires_redis
def test_cache_store_count_flush(clean_tenant_keys):
    t = _tenant()
    qvec = bytes(384 * 4)  # zero vector; only counting entries here
    store(clean_tenant_keys, t, "q1", qvec, "a1", [], corpus_version=1)
    store(clean_tenant_keys, t, "q2", qvec, "a2", [])
    assert count_entries(clean_tenant_keys, t) == 2
    removed = flush(clean_tenant_keys, t)
    assert removed == 2
    assert count_entries(clean_tenant_keys, t) == 0


@requires_redis
def test_session_append_history_clear(clean_tenant_keys):
    t = _tenant()
    append(clean_tenant_keys, t, "s1", "user", "hello")
    append(clean_tenant_keys, t, "s1", "assistant", "hi there")
    msgs = history(clean_tenant_keys, t, "s1")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "hello"
    assert clean_tenant_keys.ttl(f"{t.prefix}session:s1:msgs") > 0
    clear(clean_tenant_keys, t, "s1")
    assert history(clean_tenant_keys, t, "s1") == []


@requires_redis
def test_summary_set_get(clean_tenant_keys):
    t = _tenant()
    from rtfm_agent.common.sessions import set_summary

    set_summary(clean_tenant_keys, t, "s2", "rolling summary text")
    assert get_summary(clean_tenant_keys, t, "s2") == "rolling summary text"
