from fastapi.testclient import TestClient

from app.config import settings
from app.core import tenant_store
from app.main import app

from .conftest import random_id, requires_services


@requires_services
def test_admin_api_returns_503_when_key_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "")
    with TestClient(app) as client:
        resp = client.post(
            "/v1/admin/tenants", json={"tenant_id": "whatever"}, headers={"X-Admin-Key": "anything"}
        )
    assert resp.status_code == 503


@requires_services
def test_admin_api_rejects_wrong_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "correct-secret")
    with TestClient(app) as client:
        resp = client.post(
            "/v1/admin/tenants", json={"tenant_id": "whatever"}, headers={"X-Admin-Key": "wrong-secret"}
        )
    assert resp.status_code == 401


@requires_services
def test_admin_can_create_a_working_tenant_and_revoke_it(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "correct-secret")
    monkeypatch.setattr(tenant_store, "_DB_PATH", tmp_path / "audit.db")
    tenant_id = random_id("tenant")
    user = random_id("user")
    admin_headers = {"X-Admin-Key": "correct-secret"}

    with TestClient(app) as client:
        created = client.post("/v1/admin/tenants", json={"tenant_id": tenant_id}, headers=admin_headers)
        assert created.status_code == 200
        api_key = created.json()["api_key"]
        assert api_key  # a real, non-empty, server-generated key

        # the freshly created key must actually work against the tenant API
        write_resp = client.post(
            "/v1/memory",
            json={"text": "created via self-service admin API"},
            headers={"X-API-Key": api_key, "X-User-Id": user},
        )
        assert write_resp.status_code == 200

        usage = client.get(f"/v1/admin/tenants/{tenant_id}/usage", headers=admin_headers)
        assert usage.json()["total_memories"] >= 1

        revoke = client.delete(f"/v1/admin/tenants/{tenant_id}", headers=admin_headers)
        assert revoke.json()["revoked_keys"] >= 1

        # the revoked key must no longer authenticate
        after_revoke = client.post(
            "/v1/memory",
            json={"text": "should fail now"},
            headers={"X-API-Key": api_key, "X-User-Id": user},
        )
    assert after_revoke.status_code == 401
