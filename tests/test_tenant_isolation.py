"""Cross-tenant isolation at the pipeline/db layer, across all three stores.

Requires the docker-compose stack (qdrant, redis, neo4j) running locally —
skipped automatically otherwise. These call app.core.memory_pipeline and the
db clients directly, bypassing the API-key/header layer (covered separately
in test_api_isolation.py), to pin down the actual filtering guarantees.
"""

from app.core import memory_pipeline
from app.db import neo4j_client, qdrant_client, redis_client
from app.models.schemas import MemoryIn

from .conftest import random_id, requires_services


@requires_services
def test_qdrant_search_does_not_leak_across_tenants_with_same_user_id():
    user = random_id("user")
    tenant_a, tenant_b = random_id("tenant"), random_id("tenant")

    memory_pipeline.add_memory(tenant_a, user, MemoryIn(text="The launch codename is Falcon."))
    memory_pipeline.add_memory(tenant_b, user, MemoryIn(text="The launch codename is Falcon."))

    result_a = memory_pipeline.search_memory(tenant_a, user, "launch codename", use_graph_expansion=False)
    result_b = memory_pipeline.search_memory(tenant_b, user, "launch codename", use_graph_expansion=False)

    assert result_a.memories and result_b.memories

    ids_a = {m.memory_id for m in result_a.memories}
    ids_b = {m.memory_id for m in result_b.memories}
    assert ids_a.isdisjoint(ids_b), "same user_id under two tenants must never share memory_ids"


@requires_services
def test_qdrant_search_does_not_leak_across_users_within_same_tenant():
    tenant = random_id("tenant")
    user_a, user_b = random_id("user"), random_id("user")

    memory_pipeline.add_memory(tenant, user_a, MemoryIn(text="User A's favorite color is teal."))
    memory_pipeline.add_memory(tenant, user_b, MemoryIn(text="User B's favorite color is teal."))

    result_a = memory_pipeline.search_memory(tenant, user_a, "favorite color", use_graph_expansion=False)
    result_b = memory_pipeline.search_memory(tenant, user_b, "favorite color", use_graph_expansion=False)

    ids_a = {m.memory_id for m in result_a.memories}
    ids_b = {m.memory_id for m in result_b.memories}
    assert ids_a.isdisjoint(ids_b), "two users under the same tenant must never share memory_ids"


@requires_services
def test_redis_working_memory_isolated_per_tenant_and_user():
    user = random_id("user")
    tenant_a, tenant_b = random_id("tenant"), random_id("tenant")

    redis_client.push_turn(tenant_a, user, "user", "tenant A said this")
    redis_client.push_turn(tenant_b, user, "user", "tenant B said this")

    turns_a = redis_client.get_recent_turns(tenant_a, user)
    turns_b = redis_client.get_recent_turns(tenant_b, user)

    assert [t["text"] for t in turns_a] == ["tenant A said this"]
    assert [t["text"] for t in turns_b] == ["tenant B said this"]


@requires_services
def test_redis_frequency_counters_isolated_per_tenant():
    user = random_id("user")
    tenant_a, tenant_b = random_id("tenant"), random_id("tenant")
    memory_id = random_id("mem")

    redis_client.bump_frequency(tenant_a, user, memory_id)
    redis_client.bump_frequency(tenant_a, user, memory_id)

    assert redis_client.get_frequency(tenant_a, user, memory_id) == 2.0
    assert redis_client.get_frequency(tenant_b, user, memory_id) == 0.0


@requires_services
def test_neo4j_graph_expansion_does_not_leak_across_tenants_sharing_entity_name():
    user = random_id("user")
    tenant_a, tenant_b = random_id("tenant"), random_id("tenant")

    # Both tenants mention the same entity name ("Acme") — the composite
    # (tenant_id, user_id, name) uniqueness constraint must keep these as
    # separate graph nodes, not merge them into one shared entity.
    mem_a1 = memory_pipeline.add_memory(tenant_a, user, MemoryIn(text="Acme is our biggest customer."))
    mem_a2 = memory_pipeline.add_memory(tenant_a, user, MemoryIn(text="Acme renewed their contract."))
    mem_b1 = memory_pipeline.add_memory(tenant_b, user, MemoryIn(text="Acme is a supplier we distrust."))

    related = neo4j_client.expand_via_graph(tenant_a, user, [mem_a1])
    related_ids = {r["memory_id"] for r in related}

    assert mem_a2 in related_ids, "tenant A's own related memory should surface"
    assert mem_b1 not in related_ids, "tenant B's memory must never leak into tenant A's graph expansion"


@requires_services
def test_delete_memory_cannot_be_used_across_tenants():
    user = random_id("user")
    tenant_a, tenant_b = random_id("tenant"), random_id("tenant")

    mem_a = memory_pipeline.add_memory(tenant_a, user, MemoryIn(text="Tenant A's fact to keep."))

    # tenant B tries to delete tenant A's memory_id directly (guessed/leaked id)
    assert memory_pipeline.delete_memory(tenant_b, user, mem_a) is False

    # it must still exist for tenant A afterwards
    result = memory_pipeline.search_memory(tenant_a, user, "fact to keep", use_graph_expansion=False)
    assert mem_a in {m.memory_id for m in result.memories}


@requires_services
def test_delete_memory_removes_it_for_its_own_tenant():
    user = random_id("user")
    tenant = random_id("tenant")

    mem = memory_pipeline.add_memory(tenant, user, MemoryIn(text="Ephemeral fact to delete."))
    assert memory_pipeline.delete_memory(tenant, user, mem) is True

    result = memory_pipeline.search_memory(tenant, user, "ephemeral fact", use_graph_expansion=False)
    assert mem not in {m.memory_id for m in result.memories}


@requires_services
def test_erase_user_does_not_touch_other_tenants():
    user = random_id("user")
    tenant_a, tenant_b = random_id("tenant"), random_id("tenant")

    memory_pipeline.add_memory(tenant_a, user, MemoryIn(text="Tenant A fact, should be erased."))
    mem_b = memory_pipeline.add_memory(tenant_b, user, MemoryIn(text="Tenant B fact, must survive."))

    memory_pipeline.erase_user(tenant_a, user)

    result_a = memory_pipeline.search_memory(tenant_a, user, "fact", use_graph_expansion=False)
    result_b = memory_pipeline.search_memory(tenant_b, user, "fact", use_graph_expansion=False)

    assert result_a.memories == []
    assert mem_b in {m.memory_id for m in result_b.memories}


@requires_services
def test_update_memory_supersedes_without_deleting_history():
    user = random_id("user")
    tenant = random_id("tenant")

    v1 = memory_pipeline.add_memory(tenant, user, MemoryIn(text="Anish lives in San Francisco."))
    v2 = memory_pipeline.update_memory(tenant, user, v1, MemoryIn(text="Anish lives in Berlin now."))

    assert v2 is not None
    assert v2 != v1

    result = memory_pipeline.search_memory(tenant, user, "where does Anish live", use_graph_expansion=False)
    ids = {m.memory_id for m in result.memories}
    assert v2 in ids, "the new version should be what search returns"
    assert v1 not in ids, "the superseded version must not show up in active search"

    history = memory_pipeline.get_memory_history(tenant, user, v2)
    history_ids = [h["memory_id"] for h in history]
    assert history_ids == [v1, v2], "history should be ordered oldest -> newest"


@requires_services
def test_update_memory_cannot_be_used_across_tenants():
    user = random_id("user")
    tenant_a, tenant_b = random_id("tenant"), random_id("tenant")

    v1 = memory_pipeline.add_memory(tenant_a, user, MemoryIn(text="Tenant A's original fact."))

    # tenant B tries to supersede tenant A's memory_id directly
    result = memory_pipeline.update_memory(tenant_b, user, v1, MemoryIn(text="Hijacked by tenant B."))
    assert result is None

    # tenant A's original memory must be untouched and still active
    search = memory_pipeline.search_memory(tenant_a, user, "original fact", use_graph_expansion=False)
    assert v1 in {m.memory_id for m in search.memories}


@requires_services
def test_erase_user_removes_superseded_history_too():
    user = random_id("user")
    tenant = random_id("tenant")

    v1 = memory_pipeline.add_memory(tenant, user, MemoryIn(text="Fact to be superseded then erased."))
    v2 = memory_pipeline.update_memory(tenant, user, v1, MemoryIn(text="Replacement fact, also erased."))

    memory_pipeline.erase_user(tenant, user)

    assert qdrant_client.get_memory(v1, tenant, user) is None, "superseded version must be erased too"
    assert qdrant_client.get_memory(v2, tenant, user) is None


@requires_services
def test_neo4j_find_candidate_duplicates_scoped_to_tenant_and_user():
    user = random_id("user")
    tenant_a, tenant_b = random_id("tenant"), random_id("tenant")

    mem_a1 = memory_pipeline.add_memory(tenant_a, user, MemoryIn(text="Acme is headquartered in Berlin."))
    mem_a2 = memory_pipeline.add_memory(tenant_a, user, MemoryIn(text="Acme just opened a Berlin office."))
    mem_b = memory_pipeline.add_memory(tenant_b, user, MemoryIn(text="Acme is headquartered in Berlin."))

    # find_candidate_duplicates only looks *backward* in time from the memory
    # passed as exclude_memory_id (see its docstring) -- so check from mem_a2
    # (the newer one) to find mem_a1 (older), matching how it's actually
    # called in production: from the memory that was just written.
    candidates = neo4j_client.find_candidate_duplicates(tenant_a, user, ["Acme"], exclude_memory_id=mem_a2)
    candidate_ids = {c["memory_id"] for c in candidates}

    assert mem_a1 in candidate_ids, "tenant A's own duplicate candidate should surface"
    assert mem_a2 not in candidate_ids  # excluded itself
    assert mem_b not in candidate_ids, "tenant B's memory must never leak into tenant A's duplicate candidates"
