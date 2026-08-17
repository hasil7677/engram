from fastapi.testclient import TestClient

from app.core import tenant_store
from app.main import app

from .conftest import random_id, requires_services


@requires_services
def test_metrics_endpoint_exposes_prometheus_text_format():
    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "engram_http_requests_total" in resp.text
    assert "engram_memory_writes_total" in resp.text


@requires_services
def test_http_requests_are_labeled_by_route_template_not_raw_memory_id(tmp_path, monkeypatch):
    key = random_id("key")
    tenant_store.register_tenant_key(key, random_id("tenant"))
    user = random_id("user")

    with TestClient(app) as client:
        memory_id = client.post(
            "/v1/memory",
            json={"text": "some fact"},
            headers={"X-API-Key": key, "X-User-Id": user},
        ).json()["memory_id"]
        client.delete(f"/v1/memory/{memory_id}", headers={"X-API-Key": key, "X-User-Id": user})

        metrics_text = client.get("/metrics").text

    assert 'path_template="/v1/memory/{memory_id}"' in metrics_text
    assert memory_id not in metrics_text, "raw memory_id must never leak into a metric label"
