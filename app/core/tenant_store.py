import hashlib
import secrets
import sqlite3
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "audit.db"


def _conn() -> sqlite3.Connection:
    conn = _conn_raw()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            key_hash TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL
        )
        """
    )
    return conn


def _conn_raw() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH)


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def register_tenant_key(api_key: str, tenant_id: str) -> None:
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO api_keys (key_hash, tenant_id) VALUES (?, ?)",
        (_hash_key(api_key), tenant_id),
    )
    conn.commit()
    conn.close()


def resolve_tenant(api_key: str) -> str | None:
    """Level 1 of the two-level model: API key -> tenant_id (the developer/app).
    Never trust a client-supplied tenant_id — it must come from this lookup."""
    conn = _conn()
    row = conn.execute(
        "SELECT tenant_id FROM api_keys WHERE key_hash = ?", (_hash_key(api_key),)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def generate_and_register_tenant(tenant_id: str) -> str:
    """Server-generates a new API key for a tenant — never accepts a caller
    -supplied key, since a self-service signup endpoint must not let a caller
    pick their own (weak, guessable, or reused) secret. The plaintext key is
    returned once here; only its hash is ever stored, so it can't be
    recovered again after this call.
    """
    api_key = secrets.token_urlsafe(32)
    register_tenant_key(api_key, tenant_id)
    return api_key


def revoke_tenant(tenant_id: str) -> int:
    """Revokes every API key currently registered for this tenant_id.
    Returns how many keys were revoked."""
    conn = _conn()
    cursor = conn.execute("DELETE FROM api_keys WHERE tenant_id = ?", (tenant_id,))
    conn.commit()
    revoked = cursor.rowcount
    conn.close()
    return revoked
