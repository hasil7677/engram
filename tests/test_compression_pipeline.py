"""L2/L3/L4 memory-aging pipeline. L2/L3 need live Redis/Neo4j+Bedrock; L4's
no-op path (no S3 bucket configured) is pure logic and always runs — actually
uploading to S3 isn't covered here since that would touch a real AWS account.
"""
from datetime import date, timedelta

from app.config import settings
from app.db import neo4j_client, redis_client
from app.workers import tasks

from .conftest import random_id, requires_services


def test_archive_to_level4_is_a_noop_without_a_configured_bucket(monkeypatch):
    monkeypatch.setattr(settings, "s3_archive_bucket", "")
    tasks.archive_to_level4()  # must not raise, must not touch Neo4j/S3


@requires_services
def test_compress_to_level2_summarizes_working_memory_into_a_daily_log():
    tenant, user = random_id("tenant"), random_id("user")

    redis_client.push_turn(tenant, user, "user", "I just started powerlifting.")
    redis_client.push_turn(tenant, user, "assistant", "Nice, how's training going?")
    redis_client.push_turn(tenant, user, "user", "Good, hit a new deadlift PR today.")

    tasks.compress_to_level2()

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    rows = neo4j_client.find_stale_daily_logs(tomorrow)
    matching = [r for r in rows if r["tenant_id"] == tenant and r["user_id"] == user]

    assert matching, "compress_to_level2 should have written a DailyLog for this conversation"
    assert matching[0]["summaries"][0].strip() != ""


@requires_services
def test_compress_to_level3_rolls_up_stale_daily_logs_into_long_term_summary():
    tenant, user = random_id("tenant"), random_id("user")
    old_date = (date.today() - timedelta(days=settings.compression_l3_after_days + 5)).isoformat()

    neo4j_client.upsert_daily_log(tenant, user, old_date, "- Anish mentioned he's into powerlifting.")

    tasks.compress_to_level3()

    summary = neo4j_client.get_long_term_summary(tenant, user)
    assert summary and summary.strip() != ""

    cutoff = (date.today() + timedelta(days=1)).isoformat()
    remaining = [
        r for r in neo4j_client.find_stale_daily_logs(cutoff) if r["tenant_id"] == tenant and r["user_id"] == user
    ]
    assert remaining == [], "rolled-up daily logs should be deleted after being merged into the summary"
