"""API smoke tests through TestClient (needs live Redis; no LLM keys)."""

import pytest

from tests.conftest import REDIS_AVAILABLE

pytestmark = pytest.mark.skipif(
    not REDIS_AVAILABLE, reason="live Redis required for app lifespan"
)


@pytest.fixture(scope="module")
def client():
    """Module-scoped: the MCP session manager's lifespan runs once per process."""
    from fastapi.testclient import TestClient

    import rtfm_agent.api as api_pkg

    with TestClient(api_pkg.app) as c:
        yield c


TENANT = {"X-Tenant-Id": "apitester"}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["routing"] is True


def test_health_needs_no_tenant(client):
    assert client.get("/health").status_code == 200


def test_missing_tenant_rejected(client):
    assert client.get("/metrics").status_code == 422
    assert client.post("/ask", json={"question": "what is git?"}).status_code == 422


def test_metrics_snapshot_shape(client):
    resp = client.get("/metrics", headers=TENANT)
    assert resp.status_code == 200
    body = resp.json()
    assert {"requests_total", "hit_rate", "cache_size"} <= set(body)


def test_ask_validates_question_length(client):
    resp = client.post("/ask", json={"question": "x"}, headers=TENANT)
    assert resp.status_code == 422


def test_session_history_empty(client):
    resp = client.get("/sessions/none/history", headers=TENANT)
    assert resp.status_code == 200
    assert resp.json()["message_count"] == 0


def test_delete_missing_session_404(client):
    resp = client.delete("/sessions/nope", headers=TENANT)
    assert resp.status_code == 404
