"""End-to-end isolation through the actual HTTP layer: API key -> tenant_id
resolution, X-User-Id header handling, and that two tenants hitting the real
routes never see each other's data. Requires the docker-compose stack up
(the FastAPI lifespan provisions the Qdrant collection / Neo4j constraints).
"""

from fastapi.testclient import TestClient

from app.core import tenant_store
from app.main import app

from .conftest import random_id, requires_services


def _register(tmp_path, monkeypatch, api_key: str, tenant_id: str) -> None:
    monkeypatch.setattr(tenant_store, "_DB_PATH", tmp_path / "audit.db")
    tenant_store.register_tenant_key(api_key, tenant_id)


@requires_services
def test_two_tenants_via_api_never_see_each_others_memories(tmp_path, monkeypatch):
    key_a, key_b = random_id("key"), random_id("key")
    monkeypatch.setattr(tenant_store, "_DB_PATH", tmp_path / "audit.db")
    tenant_store.register_tenant_key(key_a, random_id("tenant"))
    tenant_store.register_tenant_key(key_b, random_id("tenant"))

    user = random_id("user")  # deliberately the same X-User-Id under both tenants

    with TestClient(app) as client:
        client.post(
            "/v1/memory",
            json={"text": "Tenant A's private fact about the user."},
            headers={"X-API-Key": key_a, "X-User-Id": user},
        )
        client.post(
            "/v1/memory",
            json={"text": "Tenant B's private fact about the user."},
            headers={"X-API-Key": key_b, "X-User-Id": user},
        )

        resp_a = client.post(
            "/v1/memory/search",
            json={"query": "private fact", "use_graph_expansion": False},
            headers={"X-API-Key": key_a, "X-User-Id": user},
        )
        resp_b = client.post(
            "/v1/memory/search",
            json={"query": "private fact", "use_graph_expansion": False},
            headers={"X-API-Key": key_b, "X-User-Id": user},
        )

    texts_a = {m["text"] for m in resp_a.json()["memories"]}
    texts_b = {m["text"] for m in resp_b.json()["memories"]}

    assert "Tenant A's private fact about the user." in texts_a
    assert "Tenant B's private fact about the user." not in texts_a
    assert "Tenant B's private fact about the user." in texts_b
    assert "Tenant A's private fact about the user." not in texts_b


@requires_services
def test_unregistered_api_key_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(tenant_store, "_DB_PATH", tmp_path / "audit.db")

    with TestClient(app) as client:
        resp = client.post(
            "/v1/memory",
            json={"text": "x"},
            headers={"X-API-Key": "not-a-real-key", "X-User-Id": "u1"},
        )
    assert resp.status_code == 401


@requires_services
def test_empty_user_id_is_rejected(tmp_path, monkeypatch):
    key = random_id("key")
    _register(tmp_path, monkeypatch, key, random_id("tenant"))

    with TestClient(app) as client:
        resp = client.post(
            "/v1/memory",
            json={"text": "x"},
            headers={"X-API-Key": key, "X-User-Id": ""},
        )
    assert resp.status_code == 400


@requires_services
def test_delete_memory_across_tenants_returns_404_and_does_not_delete(tmp_path, monkeypatch):
    key_a, key_b = random_id("key"), random_id("key")
    monkeypatch.setattr(tenant_store, "_DB_PATH", tmp_path / "audit.db")
    tenant_store.register_tenant_key(key_a, random_id("tenant"))
    tenant_store.register_tenant_key(key_b, random_id("tenant"))

    user = random_id("user")

    with TestClient(app) as client:
        memory_id = client.post(
            "/v1/memory",
            json={"text": "Tenant A's fact, must survive tenant B's delete attempt."},
            headers={"X-API-Key": key_a, "X-User-Id": user},
        ).json()["memory_id"]

        # tenant B attempts to delete tenant A's memory_id directly
        resp = client.delete(f"/v1/memory/{memory_id}", headers={"X-API-Key": key_b, "X-User-Id": user})
        assert resp.status_code == 404

        # tenant A can still find it
        search = client.post(
            "/v1/memory/search",
            json={"query": "must survive", "use_graph_expansion": False},
            headers={"X-API-Key": key_a, "X-User-Id": user},
        )
    ids = {m["memory_id"] for m in search.json()["memories"]}
    assert memory_id in ids


@requires_services
def test_update_memory_across_tenants_returns_404_and_does_not_supersede(tmp_path, monkeypatch):
    key_a, key_b = random_id("key"), random_id("key")
    monkeypatch.setattr(tenant_store, "_DB_PATH", tmp_path / "audit.db")
    tenant_store.register_tenant_key(key_a, random_id("tenant"))
    tenant_store.register_tenant_key(key_b, random_id("tenant"))

    user = random_id("user")

    with TestClient(app) as client:
        memory_id = client.post(
            "/v1/memory",
            json={"text": "Tenant A's fact, must not be superseded by tenant B."},
            headers={"X-API-Key": key_a, "X-User-Id": user},
        ).json()["memory_id"]

        resp = client.put(
            f"/v1/memory/{memory_id}",
            json={"text": "Hijacked by tenant B."},
            headers={"X-API-Key": key_b, "X-User-Id": user},
        )
        assert resp.status_code == 404

        search = client.post(
            "/v1/memory/search",
            json={"query": "must not be superseded", "use_graph_expansion": False},
            headers={"X-API-Key": key_a, "X-User-Id": user},
        )
    ids = {m["memory_id"] for m in search.json()["memories"]}
    assert memory_id in ids


@requires_services
def test_export_only_returns_this_tenant_and_users_own_records(tmp_path, monkeypatch):
    key_a, key_b = random_id("key"), random_id("key")
    monkeypatch.setattr(tenant_store, "_DB_PATH", tmp_path / "audit.db")
    tenant_store.register_tenant_key(key_a, random_id("tenant"))
    tenant_store.register_tenant_key(key_b, random_id("tenant"))

    user = random_id("user")

    with TestClient(app) as client:
        memory_id_a = client.post(
            "/v1/memory",
            json={"text": "A's fact"},
            headers={"X-API-Key": key_a, "X-User-Id": user},
        ).json()["memory_id"]
        memory_id_b = client.post(
            "/v1/memory",
            json={"text": "B's fact"},
            headers={"X-API-Key": key_b, "X-User-Id": user},
        ).json()["memory_id"]

        export_a = client.get("/v1/memory/export", headers={"X-API-Key": key_a, "X-User-Id": user})

    memory_ids_a = {r["memory_id"] for r in export_a.json()["records"]}
    assert memory_id_a in memory_ids_a
    assert memory_id_b not in memory_ids_a, "tenant A's export must never include tenant B's audit records"
