"""Level 1 of the tenant model: API key -> tenant_id. Pure sqlite, no
docker services required.
"""

from app.core import tenant_store


def test_resolve_tenant_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(tenant_store, "_DB_PATH", tmp_path / "audit.db")

    tenant_store.register_tenant_key("key-1", "tenant_alpha")
    tenant_store.register_tenant_key("key-2", "tenant_beta")

    assert tenant_store.resolve_tenant("key-1") == "tenant_alpha"
    assert tenant_store.resolve_tenant("key-2") == "tenant_beta"


def test_resolve_tenant_unknown_key_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(tenant_store, "_DB_PATH", tmp_path / "audit.db")

    assert tenant_store.resolve_tenant("never-registered") is None


def test_re_registering_key_overwrites_tenant(tmp_path, monkeypatch):
    monkeypatch.setattr(tenant_store, "_DB_PATH", tmp_path / "audit.db")

    tenant_store.register_tenant_key("shared-key", "tenant_old")
    tenant_store.register_tenant_key("shared-key", "tenant_new")

    assert tenant_store.resolve_tenant("shared-key") == "tenant_new"


def test_two_tenants_never_resolve_to_the_same_id(tmp_path, monkeypatch):
    monkeypatch.setattr(tenant_store, "_DB_PATH", tmp_path / "audit.db")

    tenant_store.register_tenant_key("key-a", "tenant_a")
    tenant_store.register_tenant_key("key-b", "tenant_b")

    assert tenant_store.resolve_tenant("key-a") != tenant_store.resolve_tenant("key-b")
