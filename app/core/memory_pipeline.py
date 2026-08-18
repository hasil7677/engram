import hashlib
import logging
import re
from datetime import datetime, timezone

from app.config import settings
from app.core import scoring
from app.core.audit import log_access
from app.core.embeddings import embed
from app.core.extraction import extract_entities_spacy
from app.core.ids import new_memory_id
from app.core.metrics import (
    memory_deletes_total,
    memory_searches_total,
    memory_updates_total,
    memory_writes_total,
    search_latency_seconds,
    semantic_cache_hits_total,
)
from app.db import neo4j_client, qdrant_client, redis_client
from app.models.schemas import MemoryIn, ScoredMemory, SearchResult
from app.workers.tasks import process_relationships_and_contradictions

logger = logging.getLogger(__name__)


def add_memory(tenant_id: str, user_id: str, memory_in: MemoryIn, supersedes: str | None = None) -> str:
    """Writes the same memory_id into all three engines, then hands off the
    expensive LLM relationship-extraction + contradiction-check to Celery.

    `supersedes`, if set, marks this as a new version replacing an older memory
    (see update_memory()) — the old memory stays in place as history, not deleted.
    """
    memory_id = new_memory_id()
    timestamp = datetime.now(timezone.utc)

    # 1. Redis: working memory (Level 1, 24h TTL)
    redis_client.push_turn(tenant_id, user_id, memory_in.role, memory_in.text)
    log_access(tenant_id, user_id, "write", "redis", memory_id, "push_turn")

    # 2. Qdrant: semantic memory
    vector = embed(memory_in.text)
    payload = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "text": memory_in.text,
        "role": memory_in.role,
        "timestamp": timestamp.isoformat(),
        "status": "active",
        "supersedes": supersedes,
        **memory_in.metadata,
    }
    qdrant_client.upsert_memory(memory_id, vector, payload)
    log_access(tenant_id, user_id, "write", "qdrant", memory_id, "upsert_memory")

    # 3. Neo4j: anchor node + fast spaCy entities (cheap, runs inline)
    neo4j_client.create_memory_node(memory_id, tenant_id, user_id, memory_in.text, timestamp, supersedes=supersedes)
    entities = extract_entities_spacy(memory_in.text)
    for ent in entities:
        neo4j_client.upsert_entity(tenant_id, user_id, ent["name"], ent["label"], timestamp)
        neo4j_client.link_memory_to_entity(memory_id, tenant_id, user_id, ent["name"])
    log_access(tenant_id, user_id, "write", "neo4j", memory_id, f"entities={len(entities)}")

    # 4. Hand off expensive work (LLM relationship mapping + contradiction check) to Celery.
    # Never block the request path on an LLM round-trip.
    process_relationships_and_contradictions.delay(
        memory_id, tenant_id, user_id, memory_in.text, entities, timestamp.isoformat()
    )

    memory_writes_total.inc()
    return memory_id


def update_memory(tenant_id: str, user_id: str, memory_id: str, memory_in: MemoryIn) -> str | None:
    """Supersedes an existing memory with a new version rather than overwriting
    it in place. The old memory is kept — marked superseded — so history/audit
    trails survive; the new memory_id becomes what search returns going forward.
    Returns None if memory_id doesn't exist or belongs to someone else.
    """
    if qdrant_client.get_memory(memory_id, tenant_id, user_id) is None:
        return None

    new_id = add_memory(tenant_id, user_id, memory_in, supersedes=memory_id)

    qdrant_client.mark_superseded(memory_id, tenant_id, user_id, superseded_by=new_id)
    neo4j_client.mark_superseded(memory_id, tenant_id, user_id, superseded_by=new_id)
    log_access(tenant_id, user_id, "update", "memory", new_id, f"supersedes={memory_id}")

    memory_updates_total.inc()
    return new_id


def get_memory_history(tenant_id: str, user_id: str, memory_id: str) -> list[dict] | None:
    """Full version chain (oldest -> newest) that memory_id belongs to.
    Returns None if memory_id doesn't exist or belongs to someone else.
    """
    if qdrant_client.get_memory(memory_id, tenant_id, user_id) is None:
        return None
    return neo4j_client.get_version_history(memory_id, tenant_id, user_id)


def delete_memory(tenant_id: str, user_id: str, memory_id: str) -> bool:
    """Deletes one memory, scoped to its owning tenant/user. Returns False if
    it doesn't exist or belongs to someone else — callers shouldn't be able
    to tell the difference between the two."""
    deleted_qdrant = qdrant_client.delete_memory(memory_id, tenant_id, user_id)
    deleted_neo4j = neo4j_client.delete_memory_node(memory_id, tenant_id, user_id)
    if deleted_qdrant or deleted_neo4j:
        log_access(tenant_id, user_id, "delete", "memory", memory_id, "single_memory_delete")
        memory_deletes_total.inc()
        return True
    return False


def erase_user(tenant_id: str, user_id: str) -> dict:
    """GDPR right to erasure: permanently removes everything this tenant/user
    has stored, across all three engines — including anything the L4 archival
    job already cold-stored in S3, via its ArchivePointer nodes. The audit log
    itself is left intact — it's the record that the erasure happened, not the
    erased data.
    """
    archived_keys = neo4j_client.find_archive_pointers(tenant_id, user_id)
    if archived_keys and settings.s3_archive_bucket:
        from app.core.s3_client import get_s3_client

        s3 = get_s3_client()
        for s3_key in archived_keys:
            try:
                s3.delete_object(Bucket=settings.s3_archive_bucket, Key=s3_key)
            except Exception:
                logger.exception("erase_user: failed to delete archived object %s", s3_key)

    qdrant_deleted = qdrant_client.delete_all_for_user(tenant_id, user_id)
    neo4j_deleted = neo4j_client.delete_all_for_user(tenant_id, user_id)
    redis_client.delete_all_for_user(tenant_id, user_id)
    log_access(
        tenant_id, user_id, "delete", "all_stores", None,
        f"erasure qdrant={qdrant_deleted} neo4j={neo4j_deleted}",
    )
    return {"qdrant_memories_deleted": qdrant_deleted, "neo4j_nodes_deleted": neo4j_deleted}


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


def search_memory(tenant_id: str, user_id: str, query: str, top_k: int = 10, use_graph_expansion: bool = True) -> SearchResult:
    memory_searches_total.inc()
    with search_latency_seconds.time():
        return _search_memory(tenant_id, user_id, query, top_k, use_graph_expansion)


def _search_memory(tenant_id: str, user_id: str, query: str, top_k: int, use_graph_expansion: bool) -> SearchResult:
    # Semantic cache: intercept repeated/near-identical queries before hitting Qdrant.
    cache_key = _query_hash(query)
    cached = redis_client.cache_get(tenant_id, user_id, cache_key)
    if cached:
        log_access(tenant_id, user_id, "read", "redis_cache", None, "semantic_cache_hit")
        semantic_cache_hits_total.inc()
        return SearchResult.model_validate_json(cached)

    vector = embed(query)
    qdrant_hits = qdrant_client.search_memory(tenant_id, user_id, vector, top_k=top_k)
    log_access(tenant_id, user_id, "read", "qdrant", None, f"query top_k={top_k}")

    seed_ids = [str(hit.id) for hit in qdrant_hits]
    max_freq = redis_client.get_max_frequency(tenant_id, user_id)

    scored: list[ScoredMemory] = []
    for hit in qdrant_hits:
        memory_id = str(hit.id)
        timestamp = datetime.fromisoformat(hit.payload["timestamp"])
        temporal = scoring.temporal_decay_score(timestamp)
        freq_raw = redis_client.get_frequency(tenant_id, user_id, memory_id)
        freq = scoring.normalized_frequency_score(freq_raw, max_freq)
        final = scoring.final_score(hit.score, temporal, freq)

        scored.append(
            ScoredMemory(
                memory_id=memory_id,
                text=hit.payload["text"],
                semantic_score=hit.score,
                temporal_score=temporal,
                frequency_score=freq,
                final_score=final,
                timestamp=timestamp,
                source="qdrant",
            )
        )

    # Graph expansion: pull in memories connected via shared entities that vector
    # search alone would miss.
    if use_graph_expansion and seed_ids:
        related = neo4j_client.expand_via_graph(tenant_id, user_id, seed_ids)
        log_access(tenant_id, user_id, "read", "neo4j", None, f"graph_expansion +{len(related)}")
        # Expanded candidates come from Neo4j, not the Qdrant search, so they have no
        # semantic score of their own — batch-fetch their stored vectors and score them
        # against the query the same way a direct hit would be, instead of a constant.
        related_vectors = qdrant_client.get_vectors([r["memory_id"] for r in related])
        for r in related:
            timestamp = datetime.fromisoformat(r["timestamp"])
            temporal = scoring.temporal_decay_score(timestamp)
            freq_raw = redis_client.get_frequency(tenant_id, user_id, r["memory_id"])
            freq = scoring.normalized_frequency_score(freq_raw, max_freq)
            related_vector = related_vectors.get(r["memory_id"])
            # fixed mid-tier fallback only if the point vanished from Qdrant between
            # the Neo4j read and this lookup — shouldn't happen, but not our invariant to break
            semantic = scoring.cosine_similarity(vector, related_vector) if related_vector else 0.5
            final = scoring.final_score(semantic, temporal, freq)
            scored.append(
                ScoredMemory(
                    memory_id=r["memory_id"],
                    text=r["text"],
                    semantic_score=semantic,
                    temporal_score=temporal,
                    frequency_score=freq,
                    final_score=final,
                    timestamp=timestamp,
                    source="graph_expansion",
                )
            )

    scored.sort(key=lambda m: m.final_score, reverse=True)
    scored = scored[:top_k]

    # Frequency should mean "this was actually returned to a caller", not "this was
    # a candidate at some point" — bump only the survivors of top_k truncation, not
    # every memory the retriever considered along the way.
    for m in scored:
        redis_client.bump_frequency(tenant_id, user_id, m.memory_id)

    context_string = build_context_string(scored)
    result = SearchResult(context_string=context_string, memories=scored)

    redis_client.cache_set(tenant_id, user_id, cache_key, result.model_dump_json())
    return result


def _sanitize_for_prompt(text: str) -> str:
    """Defence-in-depth against a stored memory hijacking the chat prompt's turn
    structure: a memory containing a newline plus "User:"/"Assistant:" can forge
    a fake turn boundary in CHAT_PROMPT_TEMPLATE, since that template has no other
    delimiter between scaffolding and retrieved content. Flattening newlines closes
    the multi-line half of that; breaking the "Role:" pattern closes the rest. This
    doesn't stop injection generally — nothing here does — it only removes the one
    structural trick that lets stored text impersonate a new turn.
    """
    flattened = " ".join(text.split())
    return re.sub(r"(?i)\b(user|assistant|system)(\s*:)", r"\1_\2", flattened)


def build_context_string(memories: list[ScoredMemory]) -> str:
    """Final output: a clean string ready to inject into an LLM prompt so the
    assistant appears to have continuous memory. Wrapped in a fence and instructed
    as data, not instructions, in CHAT_PROMPT_TEMPLATE (see app/core/chat.py) —
    mitigation, not a fix, for the injection risk documented in the README."""
    if not memories:
        return "No relevant memories found."
    lines = ["Relevant memories about this user (most relevant first):"]
    for m in memories:
        text = _sanitize_for_prompt(m.text)
        lines.append(f"- [{m.timestamp.date()}] {text} (relevance={m.final_score:.2f})")
    return "\n".join(lines)
