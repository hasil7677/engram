from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "engram_memories"

    redis_host: str = "localhost"
    redis_port: int = 6379

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "engram_dev_password"

    # Control plane (tenants, API keys, audit log) — see app/db/postgres.py for
    # why this is Postgres and not one of the three memory engines.
    database_url: str = "postgresql://engram:engram_dev_password@localhost:5432/engram"
    postgres_pool_max_size: int = 10
    postgres_pool_timeout_seconds: float = 5.0
    # How stale api_keys.last_used_at may get. Writing it on every request would
    # turn each authenticated read into a write; a minute of staleness is
    # irrelevant for "when was this key last seen" and costs ~1 write/key/min.
    api_key_last_used_throttle_seconds: int = 60

    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    bedrock_model_id: str = "mistral.mistral-7b-instruct-v0:2"
    s3_archive_bucket: str = ""  # L4 cold storage; archive_to_level4 no-ops if unset

    compression_l3_after_days: int = 30   # roll L2 daily logs into a long-term summary
    compression_l4_after_days: int = 180  # archive long-term summaries to S3

    # Contradiction-candidate discovery isn't only entity-overlap (find_candidate_duplicates)
    # -- semantically similar memories are checked too (find_semantic_candidates), since a
    # changed value ("I live in Berlin" -> "I live in Paris") shares no entity with its own
    # prior version. 0.4 empirically separates related-topic pairs (~0.47-0.78 cosine on this
    # embedding model) from unrelated ones (~-0.08-0.08).
    contradiction_semantic_threshold: float = 0.4
    contradiction_semantic_candidates: int = 5

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # all-MiniLM-L6-v2 output size

    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    rate_limit_per_minute: int = 60  # per-tenant (API key), fixed 60s window
    admin_api_key: str = ""  # gates /v1/admin/*; admin API refuses all requests if unset

    # scoring weights — tunable, not hardcoded in logic
    weight_semantic: float = 0.5
    weight_temporal: float = 0.3
    weight_frequency: float = 0.2
    temporal_decay_lambda: float = 0.02  # decay per day

    class Config:
        env_file = ".env"


settings = Settings()
