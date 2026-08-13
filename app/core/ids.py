import uuid


def new_memory_id() -> str:
    """Canonical ID shared across Qdrant payload, Redis keys, and Neo4j nodes.

    Every store must use this same id for the same fact so writes/updates/deletes
    stay reconciled across the three engines.
    """
    return str(uuid.uuid4())
