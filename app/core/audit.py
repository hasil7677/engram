import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "audit.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            action TEXT NOT NULL,
            store TEXT NOT NULL,
            memory_id TEXT,
            detail TEXT
        )
        """
    )
    return conn


def log_access(
    tenant_id: str,
    user_id: str,
    action: str,
    store: str,
    memory_id: str | None = None,
    detail: str = "",
) -> None:
    """Every DB read/write across Qdrant/Redis/Neo4j must call this.

    GDPR-driven: needed for both breach-audit trails and "export everything we hold
    on this user" data-portability requests.
    """
    conn = _conn()
    conn.execute(
        "INSERT INTO audit_log (timestamp, tenant_id, user_id, action, store, memory_id, detail) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            tenant_id,
            user_id,
            action,
            store,
            memory_id,
            detail,
        ),
    )
    conn.commit()
    conn.close()


def export_user_data(tenant_id: str, user_id: str) -> list[dict]:
    """GDPR data-portability: full access history for one user."""
    conn = _conn()
    rows = conn.execute(
        "SELECT timestamp, action, store, memory_id, detail FROM audit_log "
        "WHERE tenant_id = ? AND user_id = ? ORDER BY timestamp DESC",
        (tenant_id, user_id),
    ).fetchall()
    conn.close()
    return [
        {"timestamp": r[0], "action": r[1], "store": r[2], "memory_id": r[3], "detail": r[4]}
        for r in rows
    ]
