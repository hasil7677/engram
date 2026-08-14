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
python -m spacy download en_core_web_sm   # entity extraction; not a pip dependency
docker compose up qdrant redis neo4j      # api/celery_worker not needed
pytest
```

34 tests, all passing against live services. 5 run with no external services; the rest
exercise real cross-tenant isolation against live Qdrant/Redis/Neo4j and auto-skip cleanly if
those aren't reachable — so check for skips before reading a green run as meaningful.
`tests/test_retrieval_quality.py` is a recall@5 eval over a fixed fact/query set — the only
test that checks retrieval finds the *right* thing rather than just that isolation holds.

## Benchmark — LoCoMo, and an honest reading of it

`scripts/eval_locomo.py` runs Engram as the retriever against the official
[LoCoMo](https://arxiv.org/abs/2402.17753) dataset (ACL 2024), using the benchmark's own QA
prompt templates verbatim and its own F1 scorer. Two conversations (`conv-30` and `conv-26`,
788 dialog turns, 304 questions), answer model capped at 128 tokens.

| configuration | mean F1 | retrieval latency (median / p95) |
|---|---|---|
| top_k=5 | 0.128 | 59 ms / 109 ms |
| **top_k=20** | **0.176** | 108 ms / 315 ms |
| oracle — gold evidence, no retrieval | 0.191 | — |

**0.176 is the headline number**, at `--top-k 20`. Getting there took three experiments, two
of which refuted the hypothesis that motivated them; the sections below are in the order they
happened, because the wrong turns are the useful part.

The short version: retrieving 20 memories instead of 5 is worth +37% relative F1. It closes
three-quarters of the distance between top_k=5 and an oracle handed the correct evidence
outright, landing at 92% of that oracle's score. `top_k` was the lever the whole time — not the
token cap, and not the answer model, which is where the first two rounds of evidence pointed.

### Raising the token cap made the score worse

The earlier number here was 0.149 on `conv-30` with `max_tokens=32`, and the stated suspicion
was that truncation was destroying the score — 48 of 105 predictions ran off the end without
finishing, one of them mid-quote at `Go get '`. Raising the cap to 128 fixed exactly that:
truncation drops to 1 of 105. F1 *fell* to 0.133.

The cap was the wrong lever. LoCoMo's F1 puts precision in the denominator, gold answers
average 5 words, and Mistral-7B's average 24.5 — so letting the model write *more* lowers the
score. Truncation was a real defect, but it was cutting off prose the scorer was going to
punish anyway. Re-scoring the same predictions after mechanically keeping only the first
sentence lifts combined F1 from 0.128 to 0.138: the gap is verbosity, not knowledge.

### The oracle diagnostic

`scripts/oracle_locomo.py` removes retrieval from the loop — it builds context from exactly
the turns the dataset labels as each question's supporting `evidence`, keeping the prompt,
model, and scorer identical. It needs no running services, only Bedrock.

With perfect retrieval, combined F1 is **0.191** against 0.128 end-to-end at top_k=5. Read at
the time, that looked conclusive: the entire contribution retrieval could make was ~0.06 F1,
the ceiling was 0.19 either way, so the answer model had to be the binding constraint.

That reading was wrong, and it is worth understanding why. The oracle context is *small* —
1.3 evidence turns per question. It holds retrieval quality constant at "perfect" but also
holds context quantity at "minimal", and those two are not separable in the result. A ceiling
measured with 1.3 turns of context is not a ceiling on a retriever allowed to return 20.

The clue was already visible in the per-category split, in the two categories where the oracle
scores *worse* than ordinary retrieval:

- **Temporal.** Asked "When was Jon in Paris?" with only the gold turn as context, the model
  answers `Jon was in Paris yesterday.` — gold is `28 January 2023`. It won't resolve a
  relative reference against the session date sitting in its own context. Ordinary retrieval
  returns several memories with several date stamps, and those extra dates anchor it more often
  than the single correct turn does. Oracle temporal F1 is 0.085; at top_k=20 it is 0.192.
- **Adversarial.** These have no answer; scoring 1.0 requires abstaining. Handed a single
  plausible distractor as its entire context, the model answers instead of declining and scores
  0.014. More context gives it more chances to notice nothing supports an answer.

Both say the same thing: for this model, context quantity is doing work independent of context
precision. That is what made the oracle a floor rather than a ceiling.

Numbers above are the raw completions, exactly as the official scorer sees them. The
first-sentence figure is quoted once, explicitly labelled, and is *not* the protocol —
`oracle_locomo.py` reports it alongside the raw score purely to separate "wrong" from "wordy".

### Measuring retrieval on its own, which is what finally pointed at `top_k`

F1 and the oracle both describe the *pipeline*. Neither says whether Engram returns the right
memories, so `scripts/recall_locomo.py` measures that directly: every LoCoMo question ships
the dialog turns that support it, those turns were ingested one memory each, so a retrieved
memory maps back to a turn id by exact text match and can be checked against the gold set. No
answer model, no F1. Search-only, so it re-runs in minutes against an already-ingested
conversation.

Over the same 304 questions (381 gold evidence turns):

| cutoff | hit@k | recall@k |
|---|---|---|
| k=1 | 0.083 | 0.078 |
| **k=5** (what the eval uses) | **0.255** | **0.233** |
| k=10 | 0.371 | 0.338 |
| k=20 | 0.566 | 0.517 |

At the top_k=5 the eval had been running at, Engram surfaces at least one gold evidence turn
for only 25% of questions. The decisive detail is where the right memory lands when it misses:
of the 171 questions whose gold evidence appears anywhere in the top 20, the median rank of
the first correct hit is **7** — just past the cutoff. The relevant memories were in the index
and ranked plausibly, a handful of positions too low.

That is a ranking problem, not a storage or embedding problem, and it made the fix obvious:
raise `top_k`. Doing so lifted F1 from 0.128 to 0.176 (+37% relative), improving *every*
category, at a latency cost of median 59 → 108 ms and p95 109 → 315 ms.

| Category | n | top_k=5 | top_k=20 | oracle | recall@20 |
|---|---|---|---|---|---|
| single-hop | 114 | 0.181 | **0.233** | 0.326 | 0.605 |
| multi-hop | 43 | 0.112 | **0.205** | 0.313 | 0.325 |
| temporal | 63 | 0.139 | **0.192** | 0.085 | 0.714 |
| adversarial | 71 | 0.056 | **0.070** | 0.014 | 0.345 |
| open-domain | 13 | 0.059 | **0.070** | 0.085 | 0.318 |

So where does that leave the project? Retrieval ranking is genuinely mediocre — recall@5 of
0.233, and the score blend's weights were hand-picked and never evaluated against anything
until `recall_locomo.py` existed. Compensating with a larger `top_k` works, but it is
compensation: it buys F1 by handing the model more candidates rather than by ranking better,
and it costs ~3x p95 latency. The honest framing is that Engram's *recall* is decent and its
*precision at low k* is not, and this benchmark now has the instrument to tell those apart.

The remaining gap from 0.176 to the low-0.2s is the answer model, and that part of the earlier
conclusion stands: Mistral-7B writes 24-word prose against 5-word gold answers, and F1 punishes
that regardless of what it is handed.

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
- LoCoMo has been run on two conversations out of ten (304 of 1,986 questions). The oracle
  diagnostic now exists and says the answer model, not retrieval, is the ceiling — so the
  remaining eight conversations would sharpen the estimate without changing that conclusion.
- Ranking precision at low k is weak: recall@5 is 0.233, and the median rank of the first
  correct memory is 7. The benchmark currently compensates with `--top-k 20`, which is a
  workaround, not a fix. The score blend's weights were hand-picked and have never been tuned
  against `recall_locomo.py`, which now exists precisely to do that — that tuning is the single
  highest-value piece of work left on retrieval.
- `top_k=20` triples p95 retrieval latency (109 ms → 315 ms) and sends 20 memories into every
  prompt. Nothing here has measured the token cost of that, and for a real deployment the
  prompt-size bill would likely matter more than the latency.
- Only Mistral-7B has been tried as the answer model, so nothing here separates "Engram
  retrieves badly" from "this model answers badly" at the top end. `oracle_locomo.py --model`
  exists to settle this — it needs no services, and the oracle path is the one where the
  answer model is the only variable. The run is currently blocked outside the code: every
  Anthropic model on the Bedrock account this was built against returns
  `AccessDeniedException: INVALID_PAYMENT_INSTRUMENT`, so the comparison is one command away
  from being answerable rather than one experiment away from being designed.
- Bedrock note for anyone re-running this: on-demand invocation of current Anthropic models
  needs an *inference profile* id (`au.…` / `global.…`), not the bare model id that
  `list_foundation_models` returns. Bare ids fail with "on-demand throughput isn't supported".
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
