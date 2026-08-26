"""Tenancy validation unit tests (no Redis needed)."""

import pytest

from rtfm_agent.common.tenancy import TenantContext, normalize_tenant


@pytest.fixture(autouse=True)
def open_tenants(monkeypatch):
    """Force the allowlist open so tests don't depend on the .env."""
    from rtfm_agent.config import settings

    monkeypatch.setattr(settings.tenants, "raw", "*", raising=False)


def test_valid_slug_normalizes():
    ctx = normalize_tenant("Acme-Corp_2")
    assert ctx is not None and ctx.id == "acme-corp_2"


def test_invalid_slugs_rejected():
    for bad in ("", None, "has space", ":colon:", "x" * 100, "-lead"):
        assert normalize_tenant(bad) is None


def test_slugs_lowercased_before_validation():
    # matches the original behaviour: strip().lower() then validate
    assert normalize_tenant("UPPER") is not None


def test_context_prefixes():
    t = TenantContext("acme")
    assert t.prefix == "t:acme:"
    assert t.doc_index == "t:acme:doc_idx"
    assert t.cache_index == "t:acme:cache_idx"
    assert t.metrics_key == "t:acme:metrics:cache"


def test_repr():
    assert "acme" in repr(TenantContext("acme"))
