from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings

_client: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


def ensure_collection() -> None:
    """Single massive collection, multi-tenancy via payload (Approach B).

    Avoids per-tenant collection overhead at scale. Tenant isolation is enforced
    entirely by payload filtering in queries — see search().
    """
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in existing:
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qm.VectorParams(
            size=settings.embedding_dim,
            distance=qm.Distance.COSINE,
            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
        ),
    )
    # payload indexes so tenant/user filters are fast, not full scans
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="tenant_id",
        field_schema=qm.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="user_id",
        field_schema=qm.PayloadSchemaType.KEYWORD,
    )


def upsert_memory(memory_id: str, vector: list[float], payload: dict) -> None:
    client = get_qdrant()
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[qm.PointStruct(id=memory_id, vector=vector, payload=payload)],
    )


def tenant_filter(tenant_id: str, user_id: str) -> qm.Filter:
    """Every search MUST go through this. Never let a caller pass an unfiltered query.
    Includes every version of a memory (active or superseded) — used for erasure,
    where a full wipe must remove history too, not just the active filter().
    """
    return qm.Filter(
        must=[
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
            qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
        ]
    )


def _active_tenant_filter(tenant_id: str, user_id: str) -> qm.Filter:
    """Same as tenant_filter but excludes superseded memory versions — used by
    search so an old version doesn't show up alongside the fact that replaced it."""
    active_filter = tenant_filter(tenant_id, user_id)
    active_filter.must.append(qm.FieldCondition(key="status", match=qm.MatchValue(value="active")))
    return active_filter


def _owned_point_filter(memory_id: str, tenant_id: str, user_id: str) -> qm.Filter:
    """Scopes a single point lookup to its owning tenant/user — never trust a
    bare memory_id, since a caller could otherwise target another tenant's point."""
    return qm.Filter(
        must=[
            qm.HasIdCondition(has_id=[memory_id]),
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
            qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
        ]
    )


def search_memory(tenant_id: str, user_id: str, vector: list[float], top_k: int = 10):
    client = get_qdrant()
    return client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        query_filter=_active_tenant_filter(tenant_id, user_id),
        limit=top_k,
        with_payload=True,
    ).points


def find_semantic_candidates(
    tenant_id: str, user_id: str, vector: list[float], exclude_memory_id: str, before_timestamp: str
) -> list[dict]:
    """Contradiction-candidate discovery via semantic similarity, not shared
    entities -- catches cases like "I live in Berlin" -> "I live in Paris",
    where the changed value *is* the entity, so the two facts share nothing
    for find_candidate_duplicates (Neo4j, entity-name-keyed) to link on.

    Only returns candidates strictly older than `before_timestamp` (the new
    memory's own timestamp) -- same invariant as find_candidate_duplicates,
    and for the same reason: without it, two related facts written close
    together can each discover the other as "their" candidate and end up
    mutually superseding each other in flag_contradictions.
    """
    client = get_qdrant()
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        query_filter=_active_tenant_filter(tenant_id, user_id),
        limit=settings.contradiction_semantic_candidates + 1,  # +1: the memory itself usually matches
        score_threshold=settings.contradiction_semantic_threshold,
        with_payload=True,
    ).points
    return [
        {"memory_id": str(hit.id), "text": hit.payload["text"], "timestamp": hit.payload["timestamp"]}
        for hit in hits
        if str(hit.id) != exclude_memory_id and hit.payload["timestamp"] < before_timestamp
    ][: settings.contradiction_semantic_candidates]


def get_memory(memory_id: str, tenant_id: str, user_id: str) -> dict | None:
    """Returns the point's payload if it belongs to this tenant/user, else None."""
    client = get_qdrant()
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=_owned_point_filter(memory_id, tenant_id, user_id),
        limit=1,
        with_payload=True,
    )
    return points[0].payload if points else None


def mark_superseded(memory_id: str, tenant_id: str, user_id: str, superseded_by: str) -> None:
    """Flags a memory as replaced by a newer version rather than deleting it —
    keeps it out of active search results while preserving it as history."""
    client = get_qdrant()
    client.set_payload(
        collection_name=settings.qdrant_collection,
        payload={"status": "superseded", "superseded_by": superseded_by},
        points=qm.FilterSelector(filter=_owned_point_filter(memory_id, tenant_id, user_id)),
    )


def delete_memory(memory_id: str, tenant_id: str, user_id: str) -> bool:
    """Deletes a single memory, but only if it actually belongs to this
    tenant/user — never deletes by bare id, since a caller could otherwise
    delete another tenant's memory just by guessing/knowing a UUID."""
    client = get_qdrant()
    scoped_filter = _owned_point_filter(memory_id, tenant_id, user_id)
    existing, _ = client.scroll(collection_name=settings.qdrant_collection, scroll_filter=scoped_filter, limit=1)
    if not existing:
        return False
    client.delete(collection_name=settings.qdrant_collection, points_selector=qm.FilterSelector(filter=scoped_filter))
    return True


def count_for_tenant(tenant_id: str) -> int:
    """Total memories (all users, all versions) for one tenant — used by the
    admin usage endpoint, not by any tenant-facing route."""
    client = get_qdrant()
    tenant_only_filter = qm.Filter(must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))])
    return client.count(collection_name=settings.qdrant_collection, count_filter=tenant_only_filter).count


def delete_all_for_user(tenant_id: str, user_id: str) -> int:
    """Full erasure of one user's memories within a tenant. Returns the count deleted."""
    client = get_qdrant()
    scoped_filter = tenant_filter(tenant_id, user_id)
    count = client.count(collection_name=settings.qdrant_collection, count_filter=scoped_filter).count
    client.delete(collection_name=settings.qdrant_collection, points_selector=qm.FilterSelector(filter=scoped_filter))
    return count
