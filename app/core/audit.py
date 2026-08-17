"""GDPR audit trail: who touched what, across all three memory engines.

Was sqlite in the same file as the tenant store. Two problems, both fatal at
more than one replica: the log was per-container (so "export everything we hold
on this user" silently returned a fraction of the truth), and sqlite's
single-writer lock serialised *every* memory read and write in the API behind
one file lock, because log_access() sits in the path of all of them.
"""

from datetime import datetime, timezone

from app.db.postgres import cursor


def log_access(
    tenant_id: str,
    user_id: str,
    action: str,
    store: str,
    memory_id: str | None = None,
    detail: str = "",
) -> None:
    """Every DB read/write across Qdrant/Redis/Neo4j must call this.

    GDPR-driven: needed for both breach-audit trails and "export everything we
    hold on this user" data-portability requests.

    Written synchronously rather than queued through celery on purpose — an
    audit record that can be lost in a worker crash isn't an audit record, and
    the pooled INSERT is cheap enough that deferring it buys little.
    """
    with cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO audit_log (timestamp, tenant_id, user_id, action, store, memory_id, detail) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                datetime.now(timezone.utc),
                tenant_id,
                user_id,
                action,
                store,
                memory_id,
                detail,
            ),
        )


def export_user_data(tenant_id: str, user_id: str) -> list[dict]:
    """GDPR data-portability: full access history for one user."""
    with cursor() as cur:
        cur.execute(
            "SELECT timestamp, action, store, memory_id, detail FROM audit_log "
            "WHERE tenant_id = %s AND user_id = %s ORDER BY timestamp DESC",
            (tenant_id, user_id),
        )
        rows = cur.fetchall()
    return [
        {
            "timestamp": r[0].isoformat(),
            "action": r[1],
            "store": r[2],
            "memory_id": r[3],
            "detail": r[4],
        }
        for r in rows
    ]


def usage_for_tenant(tenant_id: str, since: datetime | None = None) -> dict[str, int]:
    """Per-tenant action counts — the billing/usage signal that deliberately
    isn't a Prometheus label (a tenant_id label would give every signup its own
    unbounded time series)."""
    query = "SELECT action, count(*) FROM audit_log WHERE tenant_id = %s"
    params: list = [tenant_id]
    if since is not None:
        query += " AND timestamp >= %s"
        params.append(since)
    query += " GROUP BY action"
    with cursor() as cur:
        cur.execute(query, tuple(params))
        return {action: count for action, count in cur.fetchall()}
