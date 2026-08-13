"""Retrieval quality eval: a fixed set of facts + queries with a known
expected match each. Nothing in the rest of the suite actually checks that
retrieval finds the *right* thing semantically — this does, and is meant to
catch regressions in the embedding model, scoring weights, or top_k handling.
"""
from app.core import memory_pipeline
from app.models.schemas import MemoryIn

from .conftest import random_id, requires_services

# (fact text, query expected to retrieve it) — topics are deliberately distinct
# so there's minimal semantic overlap between facts; a healthy retrieval layer
# should nail nearly all of these.
_FACTS_AND_QUERIES = [
    ("Anish is training for a powerlifting competition next month.", "what sport is Anish training for?"),
    ("Anish's favorite programming language is Python.", "which programming language does Anish like?"),
    ("Anish lives in Berlin, Germany.", "where does Anish live?"),
    ("Anish is allergic to peanuts.", "does Anish have any food allergies?"),
    ("Anish works at a startup building AI memory infrastructure.", "what does Anish do for work?"),
    ("Anish's dog is named Biscuit, a golden retriever.", "what is the name of Anish's pet?"),
    ("Anish prefers dark roast coffee over tea.", "what does Anish like to drink?"),
    ("Anish is learning to play the guitar in his free time.", "what instrument is Anish learning?"),
    ("Anish's favorite movie is Inception.", "what is Anish's favorite film?"),
    ("Anish drives a blue Toyota Camry.", "what car does Anish own?"),
]

RECALL_THRESHOLD = 0.8


@requires_services
def test_retrieval_recall_at_5_on_distinct_facts():
    tenant, user = random_id("tenant"), random_id("user")

    expected_ids = []
    for text, _ in _FACTS_AND_QUERIES:
        memory_id = memory_pipeline.add_memory(tenant, user, MemoryIn(text=text))
        expected_ids.append(memory_id)

    hits = 0
    misses = []
    for (text, query), expected_id in zip(_FACTS_AND_QUERIES, expected_ids):
        result = memory_pipeline.search_memory(tenant, user, query, top_k=5, use_graph_expansion=False)
        retrieved_ids = {m.memory_id for m in result.memories}
        if expected_id in retrieved_ids:
            hits += 1
        else:
            misses.append((query, text))

    recall = hits / len(_FACTS_AND_QUERIES)
    assert recall >= RECALL_THRESHOLD, (
        f"recall@5 was {recall:.2f} (threshold {RECALL_THRESHOLD}); missed queries: {misses}"
    )
