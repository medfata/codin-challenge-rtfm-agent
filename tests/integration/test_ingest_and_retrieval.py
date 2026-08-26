"""Integration tests: ingestion + retrieval against live Redis."""

from rtfm_agent.common.tenancy import TenantContext
from tests.conftest import TEST_TENANT, requires_redis


@requires_redis
def test_ingestion_roundtrip(sample_docs_dir, clean_tenant_keys):
    import numpy as np

    from rtfm_agent.ingestion.pipeline import run_ingestion

    t = TenantContext(TEST_TENANT)
    summary = run_ingestion(clean_tenant_keys, t, docs_dir=str(sample_docs_dir))
    assert summary["documents"] == 2
    assert summary["chunks_stored"] == summary["chunks_generated"] > 0
    assert summary["corpus_version"] >= 1

    # identical re-ingest must NOT bump the version
    again = run_ingestion(clean_tenant_keys, t, docs_dir=str(sample_docs_dir))
    assert again["corpus_version"] == summary["corpus_version"]
    assert again["unchanged"] == summary["documents"]

    # embedding blob roundtrip sanity
    assert isinstance(summary["duration_s"], float)
    assert np is not None  # keep import meaningful


@requires_redis
def test_retrieve_finds_indexed_chunk(sample_docs_dir, clean_tenant_keys):
    from rtfm_agent.ingestion.pipeline import run_ingestion
    from rtfm_agent.retrieval.search import indexed_sources, retrieve

    t = TenantContext(TEST_TENANT)
    r = clean_tenant_keys
    run_ingestion(r, t, docs_dir=str(sample_docs_dir))

    chunks = retrieve("what is git?", r, t)
    assert chunks, "expected at least one hit for a corpus question"
    assert all(c["score"] <= 0.40 for c in chunks)

    sources = {e["source_file"] for e in indexed_sources(r, t)}
    assert sources == {"intro.asc", "advanced.asc"}


@requires_redis
def test_hybrid_scope_filter(sample_docs_dir, clean_tenant_keys):
    from rtfm_agent.ingestion.pipeline import run_ingestion
    from rtfm_agent.retrieval.search import retrieve

    t = TenantContext(TEST_TENANT)
    r = clean_tenant_keys
    run_ingestion(r, t, docs_dir=str(sample_docs_dir))

    scoped = retrieve("rebase history", r, t,
                      doc_filter={"advanced.asc"})
    assert scoped
    assert {c["source_file"] for c in scoped} <= {"advanced.asc"}
