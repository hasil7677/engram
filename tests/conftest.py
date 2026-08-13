import socket
import uuid
from urllib.parse import urlparse

import pytest

from app.config import settings


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _services_reachable() -> bool:
    neo4j = urlparse(settings.neo4j_uri)
    return (
        _port_open(settings.qdrant_host, settings.qdrant_port)
        and _port_open(settings.redis_host, settings.redis_port)
        and _port_open(neo4j.hostname or "localhost", neo4j.port or 7687)
    )


requires_services = pytest.mark.skipif(
    not _services_reachable(),
    reason="qdrant/redis/neo4j must be reachable: run `docker compose up qdrant redis neo4j`",
)


def random_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@pytest.fixture(scope="session", autouse=True)
def _init_stores():
    """Mirrors app.main's lifespan setup, since tests call the pipeline/db
    layer directly without going through the FastAPI app startup event."""
    if not _services_reachable():
        return
    from app.db.neo4j_client import ensure_constraints
    from app.db.qdrant_client import ensure_collection

    ensure_collection()
    ensure_constraints()
