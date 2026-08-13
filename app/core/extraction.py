import json
import re
from functools import lru_cache

import spacy

from app.core.bedrock_client import invoke_chat

# spaCy label -> simplified entity label used as the graph node :label
_LABEL_MAP = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Place",
    "LOC": "Place",
    "PRODUCT": "Product",
    "EVENT": "Event",
    "WORK_OF_ART": "Work",
    "DATE": "Date",
}


@lru_cache(maxsize=1)
def get_nlp():
    return spacy.load("en_core_web_sm")


def extract_entities_spacy(text: str) -> list[dict]:
    """Fast, cheap, synchronous. Runs in the add_memory() request path."""
    doc = get_nlp()(text)
    seen = set()
    entities = []
    for ent in doc.ents:
        if ent.label_ not in _LABEL_MAP:
            continue
        name = ent.text.strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        entities.append({"name": name, "label": _LABEL_MAP[ent.label_]})
    return entities


_RELATIONSHIP_PROMPT = """You are extracting a knowledge graph from a short piece of text.

Entities already detected: {entities}

Text: "{text}"

Return ONLY a JSON array of relationships between the detected entities, in the form:
[{{"source": "EntityA", "relation": "WORKS_AT", "target": "EntityB"}}]

Rules:
- relation must be UPPER_SNAKE_CASE, max 3 words
- only use entities from the provided list (resolve aliases/pronouns to the canonical entity name)
- if no clear relationship exists, return []
- return ONLY the JSON array, no prose
"""


def _extract_json_array(raw: str) -> str:
    """Mistral-7B doesn't follow "return ONLY the JSON array" as reliably as
    Claude did — it sometimes wraps the array in a markdown fence or a sentence.
    Pull out the first top-level [...] block before parsing.
    """
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    return match.group(0) if match else raw


def map_relationships_llm(text: str, entities: list[dict]) -> list[dict]:
    """Expensive — one LLM call. Must run ONLY in the async Celery pipeline,
    never inline in the synchronous add_memory() request path.
    """
    if len(entities) < 2:
        return []

    entity_names = [e["name"] for e in entities]
    prompt = _RELATIONSHIP_PROMPT.format(entities=entity_names, text=text)

    raw = invoke_chat(prompt, max_tokens=512)

    try:
        relationships = json.loads(_extract_json_array(raw))
    except json.JSONDecodeError:
        return []

    valid_names = set(entity_names)
    return [
        r
        for r in relationships
        if r.get("source") in valid_names and r.get("target") in valid_names and r.get("relation")
    ]
