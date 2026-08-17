"""Postgres: the control-plane store (tenants, API keys, audit log).

Deliberately separate from the three memory engines. Qdrant/Redis/Neo4j hold
tenant *data*; this holds the facts that decide who a caller is and what
they're allowed to touch. Those need transactions, uniqueness constraints and
durability guarantees that none of the other three offer, and they must stay
readable when the memory engines are down — an expired key has to keep being
expired even during a Qdrant outage.

Replaces the previous sqlite file (`audit.db`), which could not survive more
than one API replica: each container got its own copy, so a key created
against one was invisible to the next, and a restart on ephemeral disk lost
every tenant.
"""

import logging
from contextlib import contextmanager

from psycopg_pool import ConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Lazily opened so importing this module never blocks on a live database —
    tests and celery workers import it without necessarily using it.
    """
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=settings.postgres_pool_max_size,
            # Fail a request that can't get a connection rather than letting it
            # hang: auth sits in front of every route, so a stuck pool here
            # would stall the whole API instead of returning a clean 503.
            timeout=settings.postgres_pool_timeout_seconds,
            open=True,
        )
    return _pool


@contextmanager
def cursor(commit: bool = False):
    """Pooled cursor. `commit=True` for writes.

    Every caller goes through the pool rather than opening its own connection:
    the audit log writes on *every* memory read and write across all three
    stores, so per-call connection setup was the single hottest avoidable cost
    in the request path.
    """
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            yield cur
        if commit:
            conn.commit()


def init_schema() -> None:
    """Idempotent DDL, run once at startup — never per request.

    Ad-hoc CREATE TABLE IF NOT EXISTS is the right weight for a two-table
    control plane, but it only survives *additive* change. The moment a column
    needs a type change, a backfill, or a rename, replace this with Alembic;
    doing it at that point is cheap, doing it after divergence between
    environments is not.
    """
    with cursor(commit=True) as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id   TEXT PRIMARY KEY,
                status      TEXT NOT NULL DEFAULT 'active',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                CONSTRAINT tenants_status_valid
                    CHECK (status IN ('active', 'suspended', 'deleted'))
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id            BIGSERIAL PRIMARY KEY,
                key_hash      TEXT NOT NULL UNIQUE,
                key_prefix    TEXT NOT NULL,
                tenant_id     TEXT NOT NULL
                                  REFERENCES tenants(tenant_id) ON DELETE CASCADE,
                name          TEXT NOT NULL DEFAULT '',
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_used_at  TIMESTAMPTZ,
                expires_at    TIMESTAMPTZ,
                revoked_at    TIMESTAMPTZ
            )
            """
        )
        # Listing a tenant's keys is the dashboard's hot path; the lookup by
        # hash is already covered by the UNIQUE constraint's implicit index.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS api_keys_tenant_idx ON api_keys (tenant_id)"
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id         BIGSERIAL PRIMARY KEY,
                timestamp  TIMESTAMPTZ NOT NULL DEFAULT now(),
                tenant_id  TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                action     TEXT NOT NULL,
                store      TEXT NOT NULL,
                memory_id  TEXT,
                detail     TEXT NOT NULL DEFAULT ''
            )
            """
        )
        # No FK to tenants on purpose: the audit trail has to outlive the
        # tenant it describes, or deleting a tenant would erase the evidence of
        # what was done to their data — exactly backwards for a breach audit.
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS audit_log_tenant_user_idx
                ON audit_log (tenant_id, user_id, timestamp DESC)
            """
        )
    logger.info("control-plane schema ready")
