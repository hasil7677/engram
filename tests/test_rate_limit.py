import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.config import settings
from app.core import tenant_store
from app.core.rate_limit import enforce_rate_limit
from app.main import app

from .conftest import random_id, requires_services


@requires_services
def test_rate_limit_blocks_after_threshold(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    tenant = random_id("tenant")

    for _ in range(3):
        enforce_rate_limit(tenant)  # within budget, must not raise

    with pytest.raises(HTTPException) as exc_info:
        enforce_rate_limit(tenant)
    assert exc_info.value.status_code == 429


@requires_services
def test_rate_limit_is_scoped_per_tenant(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    tenant_a, tenant_b = random_id("tenant"), random_id("tenant")

    enforce_rate_limit(tenant_a)
    enforce_rate_limit(tenant_a)  # tenant_a now at its limit

    enforce_rate_limit(tenant_b)  # tenant_b must be unaffected by tenant_a's usage


@requires_services
def test_api_returns_429_once_tenant_exceeds_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    key = random_id("key")
    monkeypatch.setattr(tenant_store, "_DB_PATH", tmp_path / "audit.db")
    tenant_store.register_tenant_key(key, random_id("tenant"))
    user = random_id("user")

    with TestClient(app) as client:
        headers = {"X-API-Key": key, "X-User-Id": user}
        resp1 = client.get("/v1/memory/export", headers=headers)
        resp2 = client.get("/v1/memory/export", headers=headers)
        resp3 = client.get("/v1/memory/export", headers=headers)

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp3.status_code == 429
