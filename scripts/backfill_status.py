"""One-off migration: sets status='active' on any memory written before the
versioning feature existed (those points/nodes have no status field at all,
so the active-only search filter silently excludes them).

Usage: python scripts/backfill_status.py
"""
import sys

sys.path.insert(0, ".")

from qdrant_client.http import models as qm

from app.config import settings
from app.db.neo4j_client import get_driver
from app.db.qdrant_client import get_qdrant


def backfill_qdrant() -> int:
    client = get_qdrant()
    # IsEmptyCondition matches points where the payload field is missing or null.
    missing_status = qm.Filter(must=[qm.IsEmptyCondition(is_empty=qm.PayloadField(key="status"))])

    count = client.count(collection_name=settings.qdrant_collection, count_filter=missing_status).count
    if count:
        client.set_payload(
            collection_name=settings.qdrant_collection,
            payload={"status": "active"},
            points=qm.FilterSelector(filter=missing_status),
        )
    return count


def backfill_neo4j() -> int:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (m:MemoryNode) WHERE m.status IS NULL
            SET m.status = 'active'
            RETURN count(m) AS updated
            """
        ).single()
        return result["updated"] if result else 0


if __name__ == "__main__":
    qdrant_updated = backfill_qdrant()
    neo4j_updated = backfill_neo4j()
    print(f"Qdrant: backfilled status='active' on {qdrant_updated} points")
    print(f"Neo4j: backfilled status='active' on {neo4j_updated} MemoryNodes")
