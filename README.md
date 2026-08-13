# Engram — multi-tenant memory middleware for LLM apps

A FastAPI service that stores conversational facts and hands back relevant context, so a
chatbot or agent can appear to remember a user across sessions. Three backing stores, each
doing the thing it's actually good at:

| Store | Role |
|---|---|
| **Qdrant** | Semantic memory — every fact embedded locally (`all-MiniLM-L6-v2`, 384-dim, cosine) |
| **Redis** | Working memory (last 10 turns, 24h TTL), retrieval-frequency counters, and a 5-minute semantic query cache |
| **Neo4j** | Entities (spaCy NER, in the request path) and LLM-extracted relationships (async, via Celery) — powers one-hop graph expansion at recall time |

A single `memory_id` (UUID) is the shared primary key across all three. That's the only thing
that lets updates and deletes reconcile across engines that otherwise know nothing about
each other.

**This is a learning project.** It is not a product, it has never run in production, and it
is deliberately narrow rather than general-purpose. It's public because the design decisions
below are the interesting part.

## Run it

```bash
cp .env.example .env      # fill in AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
docker compose up --build
```

Register a tenant, then write and read a memory:

```bash
docker compose exec api python scripts/register_tenant.py mykey123 tenant_demo

curl -X POST http://localhost:8000/v1/memory \
  -H "X-API-Key: mykey123" -H "X-User-Id: user_1" -H "Content-Type: application/json" \
  -d '{"text": "Anish works on Engram, a memory middleware project.", "role": "user"}'

curl -X POST http://localhost:8000/v1/memory/search \
  -H "X-API-Key: mykey123" -H "X-User-Id: user_1" -H "Content-Type: application/json" \
  -d '{"query": "what is Anish working on?"}'
```

Also available: `POST /v1/chat` (recall → prompt with context → streamed Bedrock reply →
persist the turn), `PUT /v1/memory/{id}` (supersede), `GET /v1/memory/{id}/history` (version
chain), `DELETE /v1/memory/{id}`, `DELETE /v1/memory` (full GDPR erasure),
`GET /v1/memory/export`, `GET /metrics`, and a `/v1/admin/*` surface for tenant
self-service.

There's a small React/Vite chat UI in `ui/` that talks to `/v1/chat` and shows the memories
retrieved for each reply alongside it — the fastest way to see whether the memory is actually
working, rather than inferring it from curl.

## Tests

```bash
pip install -r requirements-dev.txt
docker compose up qdrant redis neo4j    # api/celery_worker not needed
pytest
```

33 tests. 5 run with no external services; the rest exercise real cross-tenant isolation
against live Qdrant/Redis/Neo4j and auto-skip cleanly if those aren't reachable.
`tests/test_retrieval_quality.py` is a recall@5 eval over a fixed fact/query set — the only
test that checks retrieval finds the *right* thing rather than just that isolation holds.

## Benchmark — LoCoMo, and an honest reading of it

`scripts/eval_locomo.py` runs Engram as the retriever against the official
[LoCoMo](https://arxiv.org/abs/2402.17753) dataset (ACL 2024), using the benchmark's own QA
prompt templates verbatim and its own F1 scorer. One conversation so far (`conv-30`, 369
turns, 105 questions):

| | |
|---|---|
| Mean F1 | **0.149** |
| Median retrieval latency | **58 ms** |
| Mean retrieval latency | 69 ms |
| Ingest | 453 s for 369 turns |

That F1 is bad, and it's worth being precise about *why*, because the number mostly isn't
measuring retrieval:

- **The answer model is capped at 32 tokens** (`invoke_chat(prompt, max_tokens=32)`).
  Mistral-7B largely ignores "answer in the form of a short phrase" and writes prose, so
  answers get truncated mid-word — one prediction literally ends `Go get '`. F1 against a
  short gold string collapses.
- **F1 punishes verbosity even when the answer is right.** Gold `19 January, 2023` vs
  prediction `Jon lost his job as a banker in January, 2023` scores 0.25 — substantively
  correct, lexically penalised.
- Category 5 scores worst (0.083) and category 1 best (0.279), which is the shape you'd
  expect if answer synthesis, not recall, is the binding constraint.

So the honest claim is: retrieval is fast and the harness is sound; the generation half is
unturned. The next diagnostic is an oracle run — feed gold facts straight to the answer model
and re-score. If oracle F1 is also low, the answer model and token cap are the problem, not
retrieval. That experiment hasn't been run yet.

The eval deliberately calls `bedrock_client.invoke_chat` directly rather than `/v1/chat`, so
Engram's own prompt template and its `(relevance=0.70)` debug annotations don't contaminate
the comparison — an earlier run had exactly that leak into a generated answer.

## The decisions worth reading

Most of what I learned building this is in the choices, not the code:

**Multi-tenancy is two-level and was built in from day one, not retrofitted.** `tenant_id`
resolves server-side from a hashed API key — a client-supplied tenant id is never trusted.
`user_id` comes from a header and is scoped *under* the tenant. Every query in every store
filters on both. Retrofitting this later would have meant touching every call site.

**Updates never overwrite.** `PUT /v1/memory/{id}` writes a brand-new memory through the
normal add path, marks the old one `status=superseded`, and links them with a `SUPERSEDES`
edge. Search returns only active memories; superseded ones stay readable via `/history` and
are still destroyed by full erasure. This is what makes automated contradiction handling
safe — a wrong judgment costs a log entry, not data.

**Contradiction candidates come from two independent discovery paths, and only look
backward in time.** Shared entity names catch most cases, but semantic similarity is
required for the ones where the changed value *is* the entity — "I live in Berlin" → "I live
in Paris" share no entity to link on. The backward-only constraint isn't tidiness: relationship
processing runs async per write, so without it, two facts written close together can each
discover the other as *their* candidate and mutually supersede each other, vanishing the
whole pair from search instead of leaving the correct newer one standing.

**No `tenant_id` label on any Prometheus metric.** It's the obvious thing to add and it's an
unbounded-cardinality trap — every signup would mint its own time series. Per-tenant usage
lives in the audit log instead. For the same reason HTTP metrics are labeled by route
*template* (`/v1/memory/{memory_id}`), never the raw path, so a memory_id can never become a
label value.

**Deletes are ownership-scoped by construction.** `qdrant_client` verifies
`(memory_id, tenant_id, user_id)` with a filtered query before touching anything, never a
bare id. A real bug — bare-id delete with no ownership check, meaning one tenant could
delete another's memory by guessing a UUID — was caught and fixed while building this.

**Erasure has to reach into cold storage.** The L4 compression stage archives stale summaries
to S3 and leaves an `ArchivePointer` node behind. `erase_user` follows those pointers and
deletes the S3 object too, otherwise GDPR erasure would be quietly defeated by data that had
since been archived.

**The admin surface fails closed.** `/v1/admin/*` is gated by a separate `X-Admin-Key`, not
the tenant model, and returns 503 if `ADMIN_API_KEY` is unset rather than defaulting to
open. Keys are server-generated via `secrets.token_urlsafe`, returned once, and stored only
as hashes.

**Switching from Claude Haiku to Bedrock/Mistral-7B forced defensive parsing.** Mistral
follows "return ONLY JSON" and "reply with one word" far less reliably, so both LLM call
sites regex-extract the first JSON array out of possibly-prose-wrapped output and match
verdict keywords by containment, not equality. This was verified necessary against live
model output — Mistral genuinely answers `"CONTRADICTION. The new fact contradicts..."`
rather than a bare word.

**LLM calls never run in the request path.** Relationship mapping and contradiction checks
happen only in Celery. Entity NER is cheap enough to stay synchronous; anything that costs
money per call is not.

## Known gaps

Stated plainly rather than hidden:

- Celery Beat isn't wired into `docker-compose.yml`, so the L2/L3/L4 compression pipeline is
  fully implemented but never fires automatically. Running the tasks by hand works.
- LoCoMo has been run on one conversation out of ten, and the oracle diagnostic that would
  separate retrieval quality from answer-generation quality hasn't been run at all.
- `ui/README.md` is still the stock Vite template text.
- No load testing, no HA or backup story for the three stateful stores.
- No SDK — integration is raw HTTP today.
- No billing or per-tenant metering (metrics are aggregate-only by design; it would have to
  lean on the audit log).
- One pooled Qdrant collection rather than per-tenant collections. Cheaper at small scale,
  isolation enforced by payload filtering rather than physical separation. A large enough
  tenant would cause noisy-neighbour effects in the shared HNSW index.
- docker-compose is dev-only; there's no production deployment target.

## Related, but separate

[mimir](https://github.com/hasil7677/mimir) tackles the same problem domain from the opposite
direction — local-first instead of cloud-hosted, an editable Obsidian vault as the source of
truth instead of flat fact records, embedded KuZu instead of a Neo4j server. The two share no
code on purpose. Engram is where the tenant model, erasure discipline, and scoring shape were
worked out.

## License

MIT
