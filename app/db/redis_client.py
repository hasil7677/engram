import json
from datetime import datetime

import redis

from app.config import settings

_client: redis.Redis | None = None

WORKING_MEMORY_TTL_SECONDS = 24 * 60 * 60  # Level 1: 24h working memory
WORKING_MEMORY_MAX_TURNS = 10

_KNOWN_CONVERSATIONS_KEY = "known_conversations"


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis(
            host=settings.redis_host, port=settings.redis_port, decode_responses=True
        )
    return _client


def _conv_key(tenant_id: str, user_id: str) -> str:
    return f"conv:{tenant_id}:{user_id}"


def _freq_key(tenant_id: str, user_id: str) -> str:
    return f"freq:{tenant_id}:{user_id}"


def _semantic_cache_key(tenant_id: str, user_id: str, query_hash: str) -> str:
    return f"semcache:{tenant_id}:{user_id}:{query_hash}"


def push_turn(tenant_id: str, user_id: str, role: str, text: str) -> None:
    """Level 1 working memory: last N turns, List + TTL."""
    r = get_redis()
    key = _conv_key(tenant_id, user_id)
    entry = json.dumps({"role": role, "text": text, "ts": datetime.utcnow().isoformat()})
    r.lpush(key, entry)
    r.ltrim(key, 0, WORKING_MEMORY_MAX_TURNS - 1)
    r.expire(key, WORKING_MEMORY_TTL_SECONDS)
    r.sadd(_KNOWN_CONVERSATIONS_KEY, json.dumps({"tenant_id": tenant_id, "user_id": user_id}))


def get_recent_turns(tenant_id: str, user_id: str) -> list[dict]:
    r = get_redis()
    raw = r.lrange(_conv_key(tenant_id, user_id), 0, -1)
    return [json.loads(x) for x in raw]


def list_known_conversations() -> list[tuple[str, str]]:
    """Every (tenant_id, user_id) pair that has ever pushed a working-memory
    turn. Used by the L2 compression job to know what to summarize — Redis
    keys can't be safely split back into (tenant_id, user_id) since either
    could itself contain a colon, so this registry avoids parsing entirely.
    """
    r = get_redis()
    return [
        (parsed["tenant_id"], parsed["user_id"])
        for parsed in (json.loads(raw) for raw in r.smembers(_KNOWN_CONVERSATIONS_KEY))
    ]


def forget_known_conversation(tenant_id: str, user_id: str) -> None:
    r = get_redis()
    r.srem(_KNOWN_CONVERSATIONS_KEY, json.dumps({"tenant_id": tenant_id, "user_id": user_id}))


def bump_frequency(tenant_id: str, user_id: str, memory_id: str) -> None:
    """Retrieval frequency tracked via Sorted Set, score = hit count."""
    r = get_redis()
    r.zincrby(_freq_key(tenant_id, user_id), 1, memory_id)


def get_frequency(tenant_id: str, user_id: str, memory_id: str) -> float:
    r = get_redis()
    score = r.zscore(_freq_key(tenant_id, user_id), memory_id)
    return score or 0.0


def get_max_frequency(tenant_id: str, user_id: str) -> float:
    r = get_redis()
    top = r.zrevrange(_freq_key(tenant_id, user_id), 0, 0, withscores=True)
    return top[0][1] if top else 1.0


def cache_get(tenant_id: str, user_id: str, query_hash: str) -> str | None:
    r = get_redis()
    return r.get(_semantic_cache_key(tenant_id, user_id, query_hash))


def cache_set(tenant_id: str, user_id: str, query_hash: str, value: str, ttl_seconds: int = 300) -> None:
    r = get_redis()
    r.set(_semantic_cache_key(tenant_id, user_id, query_hash), value, ex=ttl_seconds)


def delete_all_for_user(tenant_id: str, user_id: str) -> None:
    """Full erasure: working memory, frequency counters, and any cached search
    results for this tenant/user. Uses SCAN (non-blocking) rather than KEYS."""
    r = get_redis()
    r.delete(_conv_key(tenant_id, user_id))
    r.delete(_freq_key(tenant_id, user_id))
    for key in r.scan_iter(match=f"semcache:{tenant_id}:{user_id}:*"):
        r.delete(key)
    forget_known_conversation(tenant_id, user_id)
