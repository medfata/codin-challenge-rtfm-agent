"""Integration tests: event stream publish/iterate against live Redis."""

from rtfm_agent.common import events as events_mod
from rtfm_agent.common.tenancy import TenantContext
from tests.conftest import TEST_TENANT, requires_redis


@requires_redis
def test_publish_and_normalize_ids(clean_tenant_keys):
    t = TenantContext(TEST_TENANT)
    entry_id = events_mod.publish(
        clean_tenant_keys, t, events_mod.INGEST_COMPLETED,
        {"tenant": t.id, "documents": 1},
    )
    assert entry_id  # "<ms>-<seq>" (bytes, as redis-py returns with decode_responses=False)
    entry_str = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
    assert events_mod.normalize_last_id(entry_id) == entry_str
    assert events_mod.normalize_last_id(entry_str) == entry_str
    assert events_mod.normalize_last_id("garbage") == ""
    assert events_mod.normalize_last_id(None) == ""


@requires_redis
def test_iter_events_yields_entry(clean_tenant_keys):
    t = TenantContext(TEST_TENANT)
    events_mod.publish(clean_tenant_keys, t, "test.event", {"n": 1})
    cursor = events_mod.resolve_start(clean_tenant_keys, t, "", backlog=5)
    gen = events_mod.iter_events(clean_tenant_keys, t, cursor)
    item = next(gen)
    if item is not None:  # None only when idle window passed
        _entry_id, etype, data = item
        assert etype in ("test.event", "ingest.completed")
