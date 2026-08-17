"""Level 1 of the tenant model: API key -> tenant_id (the developer/app).

Hashing note: keys are stored as bare SHA-256, deliberately not bcrypt/argon2.
Those exist because human-chosen passwords are low-entropy and need a slow hash
to make guessing expensive. These keys are 256 bits from `secrets.token_urlsafe`
— brute force and rainbow tables are both dead ends regardless of hash speed,
and a slow KDF would only add latency to *every authenticated request* while
making the lookup unindexable (you cannot look up a bcrypt hash by value; you'd
have to fetch and compare rows one at a time). Fast + indexed is correct here.
"""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.db.postgres import cursor

KEY_PREFIX = "eng_"
_PREFIX_DISPLAY_LEN = len(KEY_PREFIX) + 6


@dataclass(frozen=True)
class KeyResolution:
    """Why an API key did or didn't resolve.

    The caller needs the distinction: a suspended tenant is a billing problem
    the customer can fix, an unknown key is an authentication failure. Mapping
    both onto a bare `None` forced the API to answer 401 for a paid-up customer
    whose account we'd merely paused.
    """

    tenant_id: str | None
    reason: str  # ok | unknown | expired | revoked | suspended

    @property
    def ok(self) -> bool:
        return self.reason == "ok"


@dataclass(frozen=True)
class ApiKeyInfo:
    """Key metadata for the dashboard/admin API. Never carries the key itself —
    the plaintext exists only in the response that created it."""

    id: int
    tenant_id: str
    key_prefix: str
    name: str
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None

    @property
    def active(self) -> bool:
        if self.revoked_at is not None:
            return False
        return self.expires_at is None or self.expires_at > datetime.now(timezone.utc)


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def ensure_tenant(tenant_id: str) -> None:
    """Idempotent: creating a second key for an existing tenant must not reset
    that tenant's status back to active."""
    with cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO tenants (tenant_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (tenant_id,),
        )


def resolve_tenant(api_key: str) -> KeyResolution:
    """Level 1 of the two-level model. Never trust a client-supplied tenant_id —
    it must come from this lookup.

    One query answers all four failure modes, so a revoked key and a suspended
    tenant can't disagree with each other under concurrent admin action.
    """
    with cursor() as cur:
        cur.execute(
            """
            SELECT k.id, k.tenant_id, k.revoked_at, k.expires_at, t.status, k.last_used_at
              FROM api_keys k
              JOIN tenants t ON t.tenant_id = k.tenant_id
             WHERE k.key_hash = %s
            """,
            (_hash_key(api_key),),
        )
        row = cur.fetchone()

    if row is None:
        return KeyResolution(None, "unknown")

    key_id, tenant_id, revoked_at, expires_at, status, last_used_at = row
    now = datetime.now(timezone.utc)

    if revoked_at is not None:
        return KeyResolution(None, "revoked")
    if expires_at is not None and expires_at <= now:
        return KeyResolution(None, "expired")
    if status != "active":
        # tenant_id is returned even on refusal so the API can name the tenant
        # in its error and the operator can find them without a second lookup.
        return KeyResolution(tenant_id, "suspended")

    _touch_last_used(key_id, last_used_at, now)
    return KeyResolution(tenant_id, "ok")


def _touch_last_used(key_id: int, last_used_at: datetime | None, now: datetime) -> None:
    """Throttled: without this, every authenticated GET becomes a write, and a
    row lock on the key every busy tenant is sharing."""
    throttle = timedelta(seconds=settings.api_key_last_used_throttle_seconds)
    if last_used_at is not None and now - last_used_at < throttle:
        return
    with cursor(commit=True) as cur:
        cur.execute("UPDATE api_keys SET last_used_at = now() WHERE id = %s", (key_id,))


def create_key(
    tenant_id: str,
    name: str = "",
    expires_in_days: int | None = None,
) -> tuple[str, ApiKeyInfo]:
    """Issues an additional key for a tenant without disturbing existing ones.

    That "without disturbing" is the whole point: zero-downtime rotation is
    issue-new -> deploy -> revoke-old, which is impossible if a tenant can only
    ever hold one key. Returns (plaintext_key, info); the plaintext is never
    recoverable after this call.
    """
    ensure_tenant(tenant_id)
    api_key = KEY_PREFIX + secrets.token_urlsafe(32)
    expires_at = (
        datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        if expires_in_days is not None
        else None
    )
    with cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO api_keys (key_hash, key_prefix, tenant_id, name, expires_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, tenant_id, key_prefix, name, created_at,
                      last_used_at, expires_at, revoked_at
            """,
            (
                _hash_key(api_key),
                api_key[:_PREFIX_DISPLAY_LEN],
                tenant_id,
                name,
                expires_at,
            ),
        )
        info = ApiKeyInfo(*cur.fetchone())
    return api_key, info


def register_tenant_key(api_key: str, tenant_id: str, name: str = "") -> None:
    """Registers a caller-supplied key. Local/dev and migration only — the
    self-service path is `create_key`, which never lets a caller choose their
    own (weak, guessable, or already-leaked) secret.
    """
    ensure_tenant(tenant_id)
    with cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO api_keys (key_hash, key_prefix, tenant_id, name)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (key_hash) DO UPDATE
                SET tenant_id = EXCLUDED.tenant_id,
                    revoked_at = NULL
            """,
            (_hash_key(api_key), api_key[:_PREFIX_DISPLAY_LEN], tenant_id, name),
        )


def list_keys(tenant_id: str) -> list[ApiKeyInfo]:
    with cursor() as cur:
        cur.execute(
            """
            SELECT id, tenant_id, key_prefix, name, created_at,
                   last_used_at, expires_at, revoked_at
              FROM api_keys
             WHERE tenant_id = %s
             ORDER BY created_at DESC
            """,
            (tenant_id,),
        )
        return [ApiKeyInfo(*row) for row in cur.fetchall()]


def revoke_key(key_id: int) -> bool:
    """Soft revoke of a single key, so the other keys on the tenant keep working
    — the second half of a rotation. Returns False if it was already revoked."""
    with cursor(commit=True) as cur:
        cur.execute(
            "UPDATE api_keys SET revoked_at = now() "
            "WHERE id = %s AND revoked_at IS NULL",
            (key_id,),
        )
        return cur.rowcount > 0


def revoke_tenant(tenant_id: str) -> int:
    """Revokes every live key for a tenant. Returns how many were revoked.

    Soft, not DELETE: a hard delete destroys the record that the key ever
    existed, which is exactly what an incident review needs to read afterwards.
    """
    with cursor(commit=True) as cur:
        cur.execute(
            "UPDATE api_keys SET revoked_at = now() "
            "WHERE tenant_id = %s AND revoked_at IS NULL",
            (tenant_id,),
        )
        return cur.rowcount


def set_tenant_status(tenant_id: str, status: str) -> bool:
    """Suspend/reactivate without touching keys — non-payment is reversible, and
    revoking a customer's keys to pause them means they must re-integrate to
    come back. Returns False for an unknown tenant."""
    if status not in ("active", "suspended", "deleted"):
        raise ValueError(f"invalid tenant status: {status}")
    with cursor(commit=True) as cur:
        cur.execute(
            "UPDATE tenants SET status = %s WHERE tenant_id = %s", (status, tenant_id)
        )
        return cur.rowcount > 0


def get_tenant_status(tenant_id: str) -> str | None:
    with cursor() as cur:
        cur.execute("SELECT status FROM tenants WHERE tenant_id = %s", (tenant_id,))
        row = cur.fetchone()
        return row[0] if row else None
