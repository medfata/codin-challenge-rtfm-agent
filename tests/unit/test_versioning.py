"""Versioning pure-logic tests: hashing, digests, staleness messages."""

from rtfm_agent.ingestion.versioning import (
    cache_staleness_message,
    compute_digest,
    drift_changed_count,
    hash_content,
)


def test_hash_content_stable():
    assert hash_content("abc") == hash_content("abc")
    assert hash_content("abc") != hash_content("abd")


def test_compute_digest_order_independent():
    a = compute_digest({"x.asc": "h1", "y.asc": "h2"})
    b = compute_digest({"y.asc": "h2", "x.asc": "h1"})
    assert a == b


def test_digest_changes_on_any_file_change():
    base = {"x.asc": "h1", "y.asc": "h2"}
    changed = {**base, "y.asc": "h3"}
    assert compute_digest(base) != compute_digest(changed)


def test_cache_staleness_none_when_current():
    corpus = {"version": 3, "ingested_at": 0}
    assert cache_staleness_message(3, corpus) is None
    assert cache_staleness_message(0, None) is None


def test_cache_staleness_warns_when_old():
    corpus = {"version": 5, "ingested_at": 1700000000}
    msg = cache_staleness_message(4, corpus)
    assert msg and "v5" in msg and "v4" in msg


def test_cache_staleness_pre_tracking():
    corpus = {"version": 2, "ingested_at": 0}
    msg = cache_staleness_message(0, corpus)
    assert msg and "version tracking" in msg


def test_drift_changed_count():
    report = {"changed": ["a"], "added": ["b", "c"], "removed": []}
    assert drift_changed_count(report) == 3
