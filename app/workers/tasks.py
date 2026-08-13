import json
import logging
from datetime import date, datetime, timedelta, timezone

from app.config import settings
from app.core.audit import log_access
from app.core.bedrock_client import invoke_chat
from app.core.embeddings import embed
from app.core.extraction import _extract_json_array, map_relationships_llm
from app.db import neo4j_client, qdrant_client, redis_client
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.tasks.process_relationships_and_contradictions")
def process_relationships_and_contradictions(
    memory_id: str,
    tenant_id: str,
    user_id: str,
    text: str,
    entities: list[dict],
    timestamp_iso: str,
) -> None:
    """Async pipeline step run after add_memory() returns:
      1. LLM relationship mapping between entities (Entity Resolution)
      2. Contradiction / duplicate detection against candidates found two ways:
         shared entities (Neo4j) and semantic similarity (Qdrant) -- entity
         overlap alone misses cases like "I live in Berlin" -> "I live in
         Paris", where the changed value *is* the entity, so old and new
         share nothing to link on.
    Kept out of the request path because it's an LLM round-trip per fact.
    """
    timestamp = datetime.fromisoformat(timestamp_iso)

    if len(entities) >= 2:
        relationships = map_relationships_llm(text, entities)
        for rel in relationships:
            neo4j_client.link_entities(
                tenant_id, user_id, rel["source"], rel["relation"], rel["target"], timestamp
            )
        log_access(tenant_id, user_id, "write", "neo4j", memory_id, f"relationships={len(relationships)}")

    candidates_by_id: dict[str, dict] = {}

    if entities:
        entity_names = [e["name"] for e in entities]
        for c in neo4j_client.find_candidate_duplicates(tenant_id, user_id, entity_names, memory_id):
            candidates_by_id[c["memory_id"]] = c

    vector = embed(text)
    for c in qdrant_client.find_semantic_candidates(tenant_id, user_id, vector, memory_id, timestamp_iso):
        candidates_by_id.setdefault(c["memory_id"], c)

    if candidates_by_id:
        flag_contradictions.delay(memory_id, tenant_id, user_id, text, list(candidates_by_id.values()))


@celery_app.task(name="app.workers.tasks.flag_contradictions")
def flag_contradictions(memory_id: str, tenant_id: str, user_id: str, new_text: str, candidates: list[dict]) -> None:
    """Compares the new fact against memories sharing the same entities, one
    verdict per candidate via an LLM-as-judge call.

    CONTRADICTION auto-resolves: the old memory is marked superseded (Qdrant
    payload + Neo4j, same mechanism as a manual PUT /v1/memory/{id} update) so
    search/chat stop returning stale info. This is safe to automate because
    "superseded" never deletes anything — the old memory stays fully
    recoverable via GET /v1/memory/{id}/history, so a wrong LLM judgment costs
    nothing but a log entry, not data.

    DUPLICATE stays flag-and-log only, unchanged: a duplicate isn't wrong
    information, so there's no clear "old" side to retire the way there is
    with a contradiction.
    """
    if not candidates:
        return

    numbered = "\n".join(f"{i}. {c['text']}" for i, c in enumerate(candidates))
    prompt = (
        f"New fact: \"{new_text}\"\n\n"
        f"Existing related facts (indexed from 0):\n{numbered}\n\n"
        "For each existing fact, judge whether the new fact CONTRADICTS it (states something "
        "incompatible, e.g. a changed fact replacing an old one), is a DUPLICATE of it (says the "
        "same thing), or is UNRELATED to it.\n\n"
        "Return ONLY a JSON array, one entry per existing fact in the same order, in the form:\n"
        '[{"index": 0, "verdict": "CONTRADICTION"}, {"index": 1, "verdict": "UNRELATED"}]'
    )
    raw = invoke_chat(prompt, max_tokens=256)

    try:
        judgments = json.loads(_extract_json_array(raw))
    except json.JSONDecodeError:
        logger.warning("flag_contradictions: could not parse LLM verdicts for memory %s: %r", memory_id, raw)
        return

    for judgment in judgments:
        index = judgment.get("index")
        if not isinstance(index, int) or not (0 <= index < len(candidates)):
            continue
        # Mistral-7B sometimes answers with extra words around the label —
        # match on containment, not equality.
        raw_verdict = str(judgment.get("verdict", "")).upper()
        verdict = next(
            (label for label in ("CONTRADICTION", "DUPLICATE", "UNRELATED") if label in raw_verdict),
            None,
        )
        if verdict not in ("CONTRADICTION", "DUPLICATE"):
            continue

        candidate = candidates[index]
        logger.warning("memory %s flagged as %s vs candidate %s", memory_id, verdict, candidate["memory_id"])
        log_access(tenant_id, user_id, "flag", "neo4j", memory_id, f"{verdict} vs {candidate['memory_id']}")

        if verdict == "CONTRADICTION":
            qdrant_client.mark_superseded(candidate["memory_id"], tenant_id, user_id, superseded_by=memory_id)
            neo4j_client.mark_superseded(candidate["memory_id"], tenant_id, user_id, superseded_by=memory_id)
            neo4j_client.link_supersedes(memory_id, candidate["memory_id"], tenant_id, user_id)
            log_access(tenant_id, user_id, "update", "memory", memory_id, f"auto-supersedes {candidate['memory_id']}")


@celery_app.task(name="app.workers.tasks.run_compression_pipeline")
def run_compression_pipeline() -> None:
    """4-level memory aging, scaffold only:
      L1 Redis (24h TTL)          -> already expires itself, nothing to do here
      L2 Compressed 30-day logs   -> summarize L1 turns older than 24h into short logs
      L3 Long-term DB summaries   -> summarize L2 logs older than 30 days into permanent
                                      Neo4j/Qdrant summary nodes
      L4 Archival snapshots       -> cold-storage export of anything older than the
                                      long-term retention window
    Each level is its own task so failures in one don't block the others.
    """
    compress_to_level2.delay()
    compress_to_level3.delay()
    archive_to_level4.delay()


def _summarize_turns(turns: list[dict]) -> str:
    # turns come back newest-first from Redis; flip to chronological order for the prompt
    conversation = "\n".join(f"{t['role']}: {t['text']}" for t in reversed(turns))
    prompt = (
        "Summarize the key facts, preferences, and context from this conversation "
        "in 2-4 short bullet points. Be concise and factual, no commentary.\n\n"
        f"{conversation}"
    )
    return invoke_chat(prompt, max_tokens=300)


@celery_app.task(name="app.workers.tasks.compress_to_level2")
def compress_to_level2() -> None:
    """L2: summarizes each active conversation's working-memory turns into a
    durable daily log before Redis's 24h TTL silently drops them. Runs once/day
    via celery beat, so each run captures roughly a day's worth of turns.
    """
    today = date.today().isoformat()
    for tenant_id, user_id in redis_client.list_known_conversations():
        turns = redis_client.get_recent_turns(tenant_id, user_id)
        if not turns:
            # working memory already expired naturally — nothing left to compress
            redis_client.forget_known_conversation(tenant_id, user_id)
            continue

        try:
            summary = _summarize_turns(turns)
        except Exception:
            logger.exception("compress_to_level2: summarization failed for %s/%s", tenant_id, user_id)
            continue

        neo4j_client.upsert_daily_log(tenant_id, user_id, today, summary)
        log_access(tenant_id, user_id, "write", "neo4j", None, "compress_to_level2")


@celery_app.task(name="app.workers.tasks.compress_to_level3")
def compress_to_level3() -> None:
    """L3: rolls daily logs older than `compression_l3_after_days` into one
    running long-term summary per (tenant, user), then deletes the rolled-up logs."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=settings.compression_l3_after_days)).date().isoformat()

    for row in neo4j_client.find_stale_daily_logs(cutoff):
        tenant_id, user_id, summaries = row["tenant_id"], row["user_id"], row["summaries"]
        try:
            existing_summary = neo4j_client.get_long_term_summary(tenant_id, user_id)
            prompt = (
                "Merge the following notes into a single updated long-term summary of "
                "this person, in 3-6 concise bullet points. Preserve older facts unless "
                "they're clearly superseded by newer ones.\n\n"
                f"Existing summary:\n{existing_summary or '(none yet)'}\n\n"
                f"New notes:\n{chr(10).join(summaries)}"
            )
            new_summary = invoke_chat(prompt, max_tokens=400)
        except Exception:
            logger.exception("compress_to_level3: rollup failed for %s/%s", tenant_id, user_id)
            continue

        neo4j_client.upsert_long_term_summary(tenant_id, user_id, new_summary)
        neo4j_client.delete_daily_logs(tenant_id, user_id, cutoff)
        log_access(tenant_id, user_id, "write", "neo4j", None, "compress_to_level3")


@celery_app.task(name="app.workers.tasks.archive_to_level4")
def archive_to_level4() -> None:
    """L4: cold-archives long-term summaries untouched since
    `compression_l4_after_days` to S3, replacing the live node with a pointer.
    No-ops entirely if S3_ARCHIVE_BUCKET isn't configured, rather than failing
    the whole beat run over infra that may not exist in every deployment yet.
    """
    if not settings.s3_archive_bucket:
        logger.info("archive_to_level4: S3_ARCHIVE_BUCKET not set, skipping")
        return

    # imported lazily so a missing/misconfigured bucket never breaks compress_to_level2/3
    import json

    from app.core.s3_client import get_s3_client

    cutoff = (datetime.now(timezone.utc) - timedelta(days=settings.compression_l4_after_days)).isoformat()

    for row in neo4j_client.find_stale_long_term_summaries(cutoff):
        tenant_id, user_id, summary = row["tenant_id"], row["user_id"], row["summary"]
        s3_key = f"{tenant_id}/{user_id}.json"
        body = json.dumps(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "summary": summary,
                "archived_at": datetime.now(timezone.utc).isoformat(),
            }
        ).encode()

        try:
            get_s3_client().put_object(
                Bucket=settings.s3_archive_bucket, Key=s3_key, Body=body, ContentType="application/json"
            )
        except Exception:
            logger.exception("archive_to_level4: S3 upload failed for %s/%s", tenant_id, user_id)
            continue

        neo4j_client.replace_long_term_summary_with_pointer(tenant_id, user_id, s3_key)
        log_access(tenant_id, user_id, "write", "s3", None, f"archived key={s3_key}")
