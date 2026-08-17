"""Level 1 of the tenant model: API key -> tenant_id, plus key lifecycle.

Was pure sqlite and ran anywhere. Now needs a real Postgres, so it follows the
same convention as test_tenant_isolation.py: auto-skip when the service isn't
reachable. Worth the trade — the behaviour that matters here (partial revoke,
expiry, suspension) is enforced by constraints and now() in the database, so
testing it against a fake would test the fake.

Each test runs against a uniquely-named tenant and cleans up after itself, so
the suite is safe against a shared dev database.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

psycopg = pytest.importorskip("psycopg")

from app.core import tenant_store  # noqa: E402
from app.db.postgres import cursor, init_schema  # noqa: E402


def _postgres_available() -> bool:
    try:
        init_schema()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(), reason="Postgres not reachable"
)


@pytest.fixture
def tenant_id():
    tid = f"tenant_test_{uuid.uuid4().hex[:12]}"
    yield tid
    with cursor(commit=True) as cur:
        cur.execute("DELETE FROM api_keys WHERE tenant_id = %s", (tid,))
        cur.execute("DELETE FROM tenants WHERE tenant_id = %s", (tid,))


def test_created_key_resolves_to_its_tenant(tenant_id):
    api_key, info = tenant_store.create_key(tenant_id, name="primary")

    resolution = tenant_store.resolve_tenant(api_key)

    assert resolution.ok
    assert resolution.tenant_id == tenant_id
    assert info.key_prefix.startswith(tenant_store.KEY_PREFIX)
    assert info.active


def test_unknown_key_does_not_resolve():
    resolution = tenant_store.resolve_tenant("never-registered")

    assert not resolution.ok
    assert resolution.reason == "unknown"
    assert resolution.tenant_id is None


def test_two_tenants_never_resolve_to_the_same_id():
    a, b = f"t_{uuid.uuid4().hex[:8]}", f"t_{uuid.uuid4().hex[:8]}"
    try:
        key_a, _ = tenant_store.create_key(a)
        key_b, _ = tenant_store.create_key(b)

        assert tenant_store.resolve_tenant(key_a).tenant_id != (
            tenant_store.resolve_tenant(key_b).tenant_id
        )
    finally:
        with cursor(commit=True) as cur:
            cur.execute("DELETE FROM api_keys WHERE tenant_id IN (%s, %s)", (a, b))
            cur.execute("DELETE FROM tenants WHERE tenant_id IN (%s, %s)", (a, b))


def test_rotation_leaves_the_other_key_working(tenant_id):
    """The whole reason keys grew an id: revoking one must not log out the
    tenant's live integration on the other."""
    old_key, old_info = tenant_store.create_key(tenant_id, name="old")
    new_key, _ = tenant_store.create_key(tenant_id, name="new")

    assert tenant_store.revoke_key(old_info.id) is True

    assert tenant_store.resolve_tenant(old_key).reason == "revoked"
    assert tenant_store.resolve_tenant(new_key).ok


def test_revoking_twice_reports_no_change(tenant_id):
    _, info = tenant_store.create_key(tenant_id)

    assert tenant_store.revoke_key(info.id) is True
    assert tenant_store.revoke_key(info.id) is False


def test_revoke_tenant_revokes_every_live_key(tenant_id):
    key_1, _ = tenant_store.create_key(tenant_id)
    key_2, _ = tenant_store.create_key(tenant_id)

    assert tenant_store.revoke_tenant(tenant_id) == 2

    assert tenant_store.resolve_tenant(key_1).reason == "revoked"
    assert tenant_store.resolve_tenant(key_2).reason == "revoked"


def test_revocation_is_soft_so_the_record_survives(tenant_id):
    """An incident review has to be able to see that a key existed and when it
    was retired — a hard DELETE destroys exactly that evidence."""
    _, info = tenant_store.create_key(tenant_id)
    tenant_store.revoke_key(info.id)

    keys = tenant_store.list_keys(tenant_id)

    assert len(keys) == 1
    assert keys[0].revoked_at is not None
    assert keys[0].active is False


def test_expired_key_stops_resolving(tenant_id):
    api_key, info = tenant_store.create_key(tenant_id, expires_in_days=1)
    assert tenant_store.resolve_tenant(api_key).ok

    with cursor(commit=True) as cur:
        cur.execute(
            "UPDATE api_keys SET expires_at = %s WHERE id = %s",
            (datetime.now(timezone.utc) - timedelta(seconds=1), info.id),
        )

    assert tenant_store.resolve_tenant(api_key).reason == "expired"


def test_suspended_tenant_is_refused_but_distinguishable(tenant_id):
    """Suspension must not read as a bad key: the API answers 403 with the
    tenant named, not a 401 that sends a paying customer rotating keys."""
    api_key, _ = tenant_store.create_key(tenant_id)

    tenant_store.set_tenant_status(tenant_id, "suspended")

    resolution = tenant_store.resolve_tenant(api_key)
    assert resolution.reason == "suspended"
    assert resolution.tenant_id == tenant_id
    assert not resolution.ok


def test_reactivating_a_tenant_restores_existing_keys(tenant_id):
    api_key, _ = tenant_store.create_key(tenant_id)
    tenant_store.set_tenant_status(tenant_id, "suspended")

    tenant_store.set_tenant_status(tenant_id, "active")

    assert tenant_store.resolve_tenant(api_key).ok


def test_issuing_a_second_key_does_not_reactivate_a_suspended_tenant(tenant_id):
    tenant_store.create_key(tenant_id)
    tenant_store.set_tenant_status(tenant_id, "suspended")

    second_key, _ = tenant_store.create_key(tenant_id)

    assert tenant_store.resolve_tenant(second_key).reason == "suspended"


def test_invalid_status_is_rejected(tenant_id):
    tenant_store.create_key(tenant_id)

    with pytest.raises(ValueError):
        tenant_store.set_tenant_status(tenant_id, "banned")


def test_last_used_is_recorded_on_resolve(tenant_id):
    api_key, info = tenant_store.create_key(tenant_id)
    assert info.last_used_at is None

    tenant_store.resolve_tenant(api_key)

    assert tenant_store.list_keys(tenant_id)[0].last_used_at is not None


def test_keys_are_high_entropy_and_prefixed(tenant_id):
    key_1, _ = tenant_store.create_key(tenant_id)
    key_2, _ = tenant_store.create_key(tenant_id)

    assert key_1 != key_2
    assert key_1.startswith(tenant_store.KEY_PREFIX)
    # token_urlsafe(32) -> 43 chars of base64url on top of the prefix
    assert len(key_1) >= len(tenant_store.KEY_PREFIX) + 43


def test_list_keys_never_exposes_a_hash_or_plaintext(tenant_id):
    api_key, _ = tenant_store.create_key(tenant_id)

    [info] = tenant_store.list_keys(tenant_id)

    serialised = str(info)
    assert api_key not in serialised
    assert tenant_store._hash_key(api_key) not in serialised
