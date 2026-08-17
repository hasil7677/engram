"""One-shot migration of the control plane out of sqlite (`audit.db`) into Postgres.

    python scripts/migrate_sqlite_to_postgres.py            # dry run, prints counts
    python scripts/migrate_sqlite_to_postgres.py --commit   # actually writes

Moves both tables that shared audit.db:
  api_keys  (key_hash, tenant_id)  -> tenants + api_keys
  audit_log (timestamp, ...)       -> audit_log

Key hashes carry over as-is — same SHA-256 scheme — so every key already issued
to a tenant keeps working after the cutover. Nothing is deleted from the sqlite
file; keep it until Postgres has served traffic for a while.

Re-runnable: keys conflict on key_hash and are skipped, so a partial run can be
repeated. The audit log is the exception — it has no natural unique key, so
`--commit` twice would double-insert. It's skipped automatically if the target
table already holds rows, overridable with --force-audit.
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from app.db.postgres import cursor, init_schema  # noqa: E402
from app.core.tenant_store import KEY_PREFIX  # noqa: E402

_SQLITE_PATH = Path(__file__).resolve().parent.parent / "audit.db"


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def migrate_keys(conn: sqlite3.Connection, commit: bool) -> int:
    if not _table_exists(conn, "api_keys"):
        print("  api_keys: no such table in sqlite, skipping")
        return 0

    rows = conn.execute("SELECT key_hash, tenant_id FROM api_keys").fetchall()
    print(f"  api_keys: {len(rows)} row(s) found")
    if not commit:
        return len(rows)

    migrated = 0
    with cursor(commit=True) as cur:
        for key_hash, tenant_id in rows:
            cur.execute(
                "INSERT INTO tenants (tenant_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (tenant_id,),
            )
            # The plaintext is gone (only its hash was ever stored), so the
            # display prefix can't be reconstructed. A placeholder marks these
            # as pre-migration rather than pretending to show real characters.
            cur.execute(
                """
                INSERT INTO api_keys (key_hash, key_prefix, tenant_id, name)
                VALUES (%s, %s, %s, 'migrated-from-sqlite')
                ON CONFLICT (key_hash) DO NOTHING
                """,
                (key_hash, KEY_PREFIX + "legacy", tenant_id),
            )
            migrated += cur.rowcount
    return migrated


def migrate_audit(conn: sqlite3.Connection, commit: bool, force: bool) -> int:
    if not _table_exists(conn, "audit_log"):
        print("  audit_log: no such table in sqlite, skipping")
        return 0

    rows = conn.execute(
        "SELECT timestamp, tenant_id, user_id, action, store, memory_id, detail "
        "FROM audit_log ORDER BY id"
    ).fetchall()
    print(f"  audit_log: {len(rows)} row(s) found")
    if not commit:
        return len(rows)

    with cursor() as cur:
        cur.execute("SELECT count(*) FROM audit_log")
        existing = cur.fetchone()[0]
    if existing and not force:
        print(
            f"  audit_log: target already has {existing} row(s) — skipping to avoid "
            "duplicates. Re-run with --force-audit to insert anyway."
        )
        return 0

    with cursor(commit=True) as cur:
        for ts, tenant_id, user_id, action, store, memory_id, detail in rows:
            cur.execute(
                "INSERT INTO audit_log (timestamp, tenant_id, user_id, action, store, memory_id, detail) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    _parse_ts(ts),
                    tenant_id,
                    user_id,
                    action,
                    store,
                    memory_id,
                    detail or "",
                ),
            )
    return len(rows)


def _parse_ts(raw: str) -> datetime:
    """sqlite stored ISO strings; naive ones are assumed UTC, which is what
    log_access() wrote via datetime.now(timezone.utc)."""
    parsed = datetime.fromisoformat(raw)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", action="store_true", help="write to Postgres")
    parser.add_argument(
        "--force-audit",
        action="store_true",
        help="insert audit rows even if the target table is non-empty",
    )
    parser.add_argument("--sqlite-path", default=str(_SQLITE_PATH))
    args = parser.parse_args()

    path = Path(args.sqlite_path)
    if not path.exists():
        print(f"No sqlite database at {path} — nothing to migrate.")
        return 0

    print(f"Source: {path}")
    print(f"Mode:   {'COMMIT' if args.commit else 'DRY RUN (no writes)'}")

    if args.commit:
        init_schema()

    conn = sqlite3.connect(path)
    try:
        keys = migrate_keys(conn, args.commit)
        audit = migrate_audit(conn, args.commit, args.force_audit)
    finally:
        conn.close()

    verb = "migrated" if args.commit else "would migrate"
    print(f"\n{verb}: {keys} api_key(s), {audit} audit row(s)")
    if not args.commit:
        print("Re-run with --commit to apply.")
    else:
        print(f"sqlite file left untouched at {path} — delete it once you're confident.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
