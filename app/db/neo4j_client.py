from datetime import datetime, timezone

from neo4j import GraphDatabase

from app.config import settings

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
    return _driver


def ensure_constraints() -> None:
    driver = get_driver()
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT entity_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.tenant_id, e.user_id, e.name) IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT memory_node_unique IF NOT EXISTS "
            "FOR (m:MemoryNode) REQUIRE m.memory_id IS UNIQUE"
        )


def create_memory_node(
    memory_id: str, tenant_id: str, user_id: str, text: str, timestamp: datetime, supersedes: str | None = None
) -> None:
    """Anchor node for a raw memory — same memory_id as Qdrant/Redis.
    `supersedes`, if set, means this node is a new version replacing an older
    memory — a SUPERSEDES edge is added and the old node stays as history.
    """
    driver = get_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (m:MemoryNode {memory_id: $memory_id})
            SET m.tenant_id = $tenant_id,
                m.user_id = $user_id,
                m.text = $text,
                m.timestamp = $timestamp,
                m.status = 'active',
                m.supersedes = $supersedes
            """,
            memory_id=memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
            text=text,
            timestamp=timestamp.isoformat(),
            supersedes=supersedes,
        )
        if supersedes:
            session.run(
                """
                MATCH (new:MemoryNode {memory_id: $memory_id})
                MATCH (old:MemoryNode {memory_id: $supersedes, tenant_id: $tenant_id, user_id: $user_id})
                MERGE (new)-[:SUPERSEDES]->(old)
                """,
                memory_id=memory_id,
                supersedes=supersedes,
                tenant_id=tenant_id,
                user_id=user_id,
            )


def mark_superseded(memory_id: str, tenant_id: str, user_id: str, superseded_by: str) -> None:
    """Flags the old version as replaced — the SUPERSEDES edge itself is
    created by create_memory_node() when the new version is written."""
    driver = get_driver()
    with driver.session() as session:
        session.run(
            """
            MATCH (m:MemoryNode {memory_id: $memory_id, tenant_id: $tenant_id, user_id: $user_id})
            SET m.status = 'superseded', m.superseded_by = $superseded_by
            """,
            memory_id=memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
            superseded_by=superseded_by,
        )


def link_supersedes(new_memory_id: str, old_memory_id: str, tenant_id: str, user_id: str) -> None:
    """Adds a SUPERSEDES edge after the fact -- for when a memory is only
    discovered to replace another one later (async contradiction resolution),
    unlike the normal update_memory() path where create_memory_node() already
    knows the relationship at write time."""
    driver = get_driver()
    with driver.session() as session:
        session.run(
            """
            MATCH (new:MemoryNode {memory_id: $new_memory_id, tenant_id: $tenant_id, user_id: $user_id})
            MATCH (old:MemoryNode {memory_id: $old_memory_id, tenant_id: $tenant_id, user_id: $user_id})
            MERGE (new)-[:SUPERSEDES]->(old)
            """,
            new_memory_id=new_memory_id,
            old_memory_id=old_memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )


def get_version_history(memory_id: str, tenant_id: str, user_id: str) -> list[dict]:
    """Full supersede chain this memory_id belongs to (oldest -> newest),
    regardless of which version in the chain you start from."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (start:MemoryNode {memory_id: $memory_id, tenant_id: $tenant_id, user_id: $user_id})
            OPTIONAL MATCH (start)-[:SUPERSEDES*]->(older:MemoryNode)
            OPTIONAL MATCH (newer:MemoryNode)-[:SUPERSEDES*]->(start)
            WITH start, collect(DISTINCT older) AS olders, collect(DISTINCT newer) AS newers
            UNWIND olders + [start] + newers AS m
            RETURN DISTINCT m.memory_id AS memory_id, m.text AS text,
                   m.timestamp AS timestamp, m.status AS status
            ORDER BY timestamp ASC
            """,
            memory_id=memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        return [dict(r) for r in result]


def upsert_entity(tenant_id: str, user_id: str, name: str, label: str, timestamp: datetime) -> None:
    driver = get_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (e:Entity {tenant_id: $tenant_id, user_id: $user_id, name: $name})
            ON CREATE SET e.label = $label, e.timestamp = $timestamp
            ON MATCH SET e.last_seen = $timestamp
            """,
            tenant_id=tenant_id,
            user_id=user_id,
            name=name,
            label=label,
            timestamp=timestamp.isoformat(),
        )


def link_memory_to_entity(memory_id: str, tenant_id: str, user_id: str, entity_name: str) -> None:
    driver = get_driver()
    with driver.session() as session:
        session.run(
            """
            MATCH (m:MemoryNode {memory_id: $memory_id})
            MATCH (e:Entity {tenant_id: $tenant_id, user_id: $user_id, name: $entity_name})
            MERGE (m)-[:MENTIONS]->(e)
            """,
            memory_id=memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
            entity_name=entity_name,
        )


def link_entities(
    tenant_id: str, user_id: str, source: str, relation: str, target: str, timestamp: datetime
) -> None:
    """Relationship edges produced by the LLM relationship-mapping step."""
    driver = get_driver()
    with driver.session() as session:
        session.run(
            f"""
            MATCH (a:Entity {{tenant_id: $tenant_id, user_id: $user_id, name: $source}})
            MATCH (b:Entity {{tenant_id: $tenant_id, user_id: $user_id, name: $target}})
            MERGE (a)-[r:`{relation}`]->(b)
            SET r.timestamp = $timestamp
            """,
            tenant_id=tenant_id,
            user_id=user_id,
            source=source,
            target=target,
            timestamp=timestamp.isoformat(),
        )


def get_memory_timestamp(memory_id: str) -> datetime | None:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            "MATCH (m:MemoryNode {memory_id: $memory_id}) RETURN m.timestamp AS ts",
            memory_id=memory_id,
        ).single()
        return datetime.fromisoformat(result["ts"]) if result else None


def expand_via_graph(tenant_id: str, user_id: str, memory_ids: list[str], hops: int = 1) -> list[dict]:
    """Graph expansion: from seed memory nodes, walk shared entities to find related
    memories that pure vector search wouldn't surface."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (seed:MemoryNode)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:MemoryNode)
            WHERE seed.memory_id IN $memory_ids
              AND related.tenant_id = $tenant_id AND related.user_id = $user_id
              AND NOT related.memory_id IN $memory_ids
              AND related.status <> 'superseded'
            RETURN DISTINCT related.memory_id AS memory_id, related.text AS text, related.timestamp AS timestamp
            LIMIT 20
            """,
            memory_ids=memory_ids,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        return [dict(r) for r in result]


def delete_memory_node(memory_id: str, tenant_id: str, user_id: str) -> bool:
    """Deletes one memory's anchor node (and its MENTIONS edges), scoped to the
    caller's tenant/user so a memory_id from another tenant can't be targeted."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (m:MemoryNode {memory_id: $memory_id, tenant_id: $tenant_id, user_id: $user_id})
            DETACH DELETE m
            RETURN count(m) AS deleted
            """,
            memory_id=memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
        ).single()
        return bool(result and result["deleted"] > 0)


def delete_all_for_user(tenant_id: str, user_id: str) -> int:
    """Full erasure: removes every MemoryNode and Entity scoped to this tenant/user."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n) WHERE n.tenant_id = $tenant_id AND n.user_id = $user_id
            DETACH DELETE n
            RETURN count(n) AS deleted
            """,
            tenant_id=tenant_id,
            user_id=user_id,
        ).single()
        return result["deleted"] if result else 0


def find_candidate_duplicates(tenant_id: str, user_id: str, entity_names: list[str], exclude_memory_id: str) -> list[dict]:
    """Used by the async contradiction/consolidation task to find memories that
    touch the same entities as a newly ingested fact.

    Only looks *backward* in time (candidate.timestamp < new memory's own
    timestamp), matched via the new memory's own node rather than a passed-in
    timestamp. This is required, not just tidy: process_relationships_and_contradictions
    runs async per write, so if two related facts are written close together,
    each one's Celery task can end up running after *both* memory nodes
    already exist in Neo4j — without the time ordering, each task would find
    the other as a candidate and they'd flag (and, with CONTRADICTION,
    auto-supersede) each other, silently vanishing the whole pair from active
    search instead of leaving the correct newer one standing.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (new:MemoryNode {memory_id: $exclude_memory_id, tenant_id: $tenant_id, user_id: $user_id})
            MATCH (m:MemoryNode)-[:MENTIONS]->(e:Entity)
            WHERE e.name IN $entity_names
              AND m.tenant_id = $tenant_id AND m.user_id = $user_id
              AND m.memory_id <> $exclude_memory_id
              AND m.status <> 'superseded'
              AND m.timestamp < new.timestamp
            RETURN DISTINCT m.memory_id AS memory_id, m.text AS text, m.timestamp AS timestamp
            LIMIT 20
            """,
            entity_names=entity_names,
            tenant_id=tenant_id,
            user_id=user_id,
            exclude_memory_id=exclude_memory_id,
        )
        return [dict(r) for r in result]


# --- L2/L3/L4 memory aging (see app.workers.tasks) ---


def upsert_daily_log(tenant_id: str, user_id: str, date_str: str, summary: str) -> None:
    """L2: one compressed summary per (tenant, user, day) — re-running the same
    day updates it rather than duplicating."""
    driver = get_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (d:DailyLog {tenant_id: $tenant_id, user_id: $user_id, date: $date_str})
            SET d.summary = $summary, d.updated_at = $now
            """,
            tenant_id=tenant_id,
            user_id=user_id,
            date_str=date_str,
            summary=summary,
            now=datetime.now(timezone.utc).isoformat(),
        )


def find_stale_daily_logs(cutoff_date: str) -> list[dict]:
    """Groups DailyLogs older than cutoff_date by (tenant_id, user_id) — Cypher
    groups automatically on the non-aggregated RETURN fields alongside collect()."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:DailyLog) WHERE d.date < $cutoff_date
            RETURN d.tenant_id AS tenant_id, d.user_id AS user_id, collect(d.summary) AS summaries
            """,
            cutoff_date=cutoff_date,
        )
        return [dict(r) for r in result]


def get_long_term_summary(tenant_id: str, user_id: str) -> str | None:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            "MATCH (s:LongTermSummary {tenant_id: $tenant_id, user_id: $user_id}) RETURN s.summary AS summary",
            tenant_id=tenant_id,
            user_id=user_id,
        ).single()
        return result["summary"] if result else None


def upsert_long_term_summary(tenant_id: str, user_id: str, summary: str) -> None:
    """L3: one running summary per (tenant, user) — each rollup merges new
    daily logs into whatever summary already exists, then callers delete the
    rolled-up daily logs (see delete_daily_logs)."""
    driver = get_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (s:LongTermSummary {tenant_id: $tenant_id, user_id: $user_id})
            SET s.summary = $summary, s.updated_at = $now
            """,
            tenant_id=tenant_id,
            user_id=user_id,
            summary=summary,
            now=datetime.now(timezone.utc).isoformat(),
        )


def delete_daily_logs(tenant_id: str, user_id: str, cutoff_date: str) -> int:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:DailyLog {tenant_id: $tenant_id, user_id: $user_id}) WHERE d.date < $cutoff_date
            DETACH DELETE d
            RETURN count(d) AS deleted
            """,
            tenant_id=tenant_id,
            user_id=user_id,
            cutoff_date=cutoff_date,
        ).single()
        return result["deleted"] if result else 0


def find_stale_long_term_summaries(cutoff_iso: str) -> list[dict]:
    """L4 candidates: long-term summaries not updated since cutoff_iso."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:LongTermSummary) WHERE s.updated_at < $cutoff
            RETURN s.tenant_id AS tenant_id, s.user_id AS user_id, s.summary AS summary
            """,
            cutoff=cutoff_iso,
        )
        return [dict(r) for r in result]


def replace_long_term_summary_with_pointer(tenant_id: str, user_id: str, s3_key: str) -> None:
    """L4: the live summary node is replaced by a lightweight pointer to where
    the content actually lives now (cold storage) — erase_user() uses this
    pointer to also delete the archived object on full GDPR erasure."""
    driver = get_driver()
    with driver.session() as session:
        session.run(
            "MATCH (s:LongTermSummary {tenant_id: $tenant_id, user_id: $user_id}) DETACH DELETE s",
            tenant_id=tenant_id,
            user_id=user_id,
        )
        session.run(
            """
            MERGE (p:ArchivePointer {tenant_id: $tenant_id, user_id: $user_id})
            SET p.s3_key = $s3_key, p.archived_at = $now
            """,
            tenant_id=tenant_id,
            user_id=user_id,
            s3_key=s3_key,
            now=datetime.now(timezone.utc).isoformat(),
        )


def find_archive_pointers(tenant_id: str, user_id: str) -> list[str]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            "MATCH (p:ArchivePointer {tenant_id: $tenant_id, user_id: $user_id}) RETURN p.s3_key AS s3_key",
            tenant_id=tenant_id,
            user_id=user_id,
        )
        return [r["s3_key"] for r in result]
