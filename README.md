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
docker compose exec api python scripts/register_tenant.py tenant_demo mykey123

curl -X POST http://localhost:8000/v1/memory \
  -H "X-API-Key: mykey123" -H "X-User-Id: user_1" -H "Content-Type: application/json" \
  -d '{"text": "Anish works on Engram, a memory middleware project.", "role": "user"}'

curl -X POST http://localhost:8000/v1/memory/search \
  -H "X-API-Key: mykey123" -H "X-User-Id: user_1" -H "Content-Type: application/json" \
  -d '{"query": "what is Anish working on?"}'
```

Drop the `mykey123` argument to get a real server-generated key instead — the
fixed-key form is a local convenience so the curl examples above stay copy-pasteable.

Upgrading an existing checkout? Tenants, API keys and the audit log used to live in a
local `audit.db` sqlite file and now live in Postgres. Move them across before first
boot; existing keys keep working, since the hash scheme is unchanged:

```bash
python scripts/migrate_sqlite_to_postgres.py            # dry run, prints counts
python scripts/migrate_sqlite_to_postgres.py --commit
```

### Key rotation and suspension

API keys are per-tenant *rows*, not one-per-tenant, which is what makes rotation
zero-downtime: issue → deploy → retire, with both keys valid in the middle.

```bash
# 1. issue a second key; the current one keeps working
curl -X POST http://localhost:8000/v1/admin/tenants/tenant_demo/keys \
  -H "X-Admin-Key: <ADMIN_API_KEY>" -H "Content-Type: application/json" \
  -d '{"name": "rotation-2026-08", "expires_in_days": 90}'

# 2. deploy the new key, then check the old one has actually gone quiet
curl http://localhost:8000/v1/admin/tenants/tenant_demo/keys -H "X-Admin-Key: <ADMIN_API_KEY>"
#    -> per key: prefix, name, created_at, last_used_at, expires_at, active

# 3. retire the old one by id
curl -X DELETE http://localhost:8000/v1/admin/tenants/tenant_demo/keys/<key_id> \
  -H "X-Admin-Key: <ADMIN_API_KEY>"
```

Suspension is separate from revocation, and reversible — pausing a tenant leaves their
keys intact, so reactivating resumes service with no re-integration:

```bash
curl -X PUT http://localhost:8000/v1/admin/tenants/tenant_demo/status \
  -H "X-Admin-Key: <ADMIN_API_KEY>" -H "Content-Type: application/json" \
  -d '{"status": "suspended"}'   # active | suspended | deleted
```

Revocation is soft throughout: a revoked key is marked, never deleted, because the
record that a key existed and when it was retired is exactly what an incident review
needs to read afterwards.

### Security boundary — the one thing to read before wrapping this in an SDK

`X-API-Key` identifies the **tenant** (the developer). `X-User-Id` identifies an
end-user *inside* that tenant, and is **asserted by the caller, not authenticated**.
That's the right design for a server-side SDK, where the tenant's own backend is
trusted. It also means:

> **The API key must never ship in a browser, a mobile app, or any client you don't
> control.** Anyone holding it can pass any `X-User-Id` and read every end-user's
> memories under that tenant.

Client-side use needs short-lived per-user tokens minted by the tenant's backend.
That doesn't exist here — it's the gate on a browser SDK, not a detail to add later.

### Prompt injection via stored memories — real, and only partially mitigated

Retrieved memory text is interpolated into the chat prompt built for `/v1/chat`
(`build_context_string` in `app/core/memory_pipeline.py`, consumed by
`CHAT_PROMPT_TEMPLATE` in `app/core/chat.py`). Memories are fenced with delimiters
and have newlines/turn markers stripped before insertion (see `_sanitize_for_prompt`
in `memory_pipeline.py`) — enough to stop a memory from forging a fake `User:`/
`Assistant:` turn boundary, but fencing is defence-in-depth, not a solved problem:
a sufficiently creative payload inside the fence can still try to talk the model
into ignoring the fence's own instructions.

It's a *persistent* injection risk, not a one-shot one: `stream_chat_reply` persists
both turns of every chat exchange, so a user's own message becomes a stored memory
that's retrieved into every future prompt. Say the adversarial thing once and it's
durable. It also propagates into the L2/L3 compression summaries, since those
prompts are built from the same memory text — a payload can get laundered into a
long-term summary, where it's harder to spot.

Blast radius connects to the boundary above: since `X-User-Id` is asserted, not
authenticated, anyone holding a tenant's API key can write memories to *any*
`user_id` under that tenant — so a compromised key can poison an arbitrary user's
memory, not just their own.

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
docker compose up qdrant redis neo4j postgres   # api/celery_worker not needed
pytest
```

Every test now needs live services and auto-skips cleanly when they aren't reachable — so
check for skips before reading a green run as meaningful. `tests/test_tenant_store.py` used
to be the exception (pure sqlite, ran anywhere); it moved to Postgres along with the store
itself, and that was the right trade, because what it asserts — partial revoke, expiry,
suspension — is enforced by database constraints and `now()`, so testing it against a fake
would only have tested the fake.
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

*(This F1 table predates the bump-timing and graph-expansion fixes described below in "Re-measured
after fixing..." — those change ranking, not retrieval count, so the qualitative story here should
hold, but the exact numbers haven't been re-run through the full answer-model pipeline.)*

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

Raising `top_k` is compensation, not a fix: it buys F1 by handing the model more candidates
rather than by ranking better, and it costs ~3x p95 latency. So the next question was what the
ranking is actually doing wrong.

### The score blend, measured — the frequency term is the bug

`recall_locomo.py` records the three score components per returned memory, not just the final
score. Since the blend is a weighted sum, one search pass at `--top-k 50` makes every possible
weighting testable offline: recompute, re-sort, re-score. Over the same 304 questions:

| weights (semantic / temporal / frequency) | recall@5 |
|---|---|
| **0.5 / 0.3 / 0.2** — as shipped | 0.183 |
| 0.5 / **0.0** / 0.2 — temporal removed | 0.183 |
| 0.5 / 0.3 / **0.0** — frequency removed | **0.384** |
| 1.0 / 0 / 0 — semantic only | 0.384 |

**Removing the frequency term more than doubles recall@5, and removing the temporal term
changes nothing at all** — not approximately nothing, exactly nothing. The median rank of the
first correct memory goes from 13 to 4.

Both results have the same cause, visible in the component distributions across all 15,200
retrieved candidates:

- **Temporal is inert here.** It ranges 0.9802 to 0.9888 — every LoCoMo memory was ingested
  within the same few minutes, so temporal decay adds a near-identical ~0.295 to every score
  and cannot reorder anything. That is a property of the benchmark, not a defect: `temporal_decay_score`
  keys off *ingestion* time, and this corpus has no ingestion-time spread. **This benchmark
  cannot evaluate temporal weighting, so none of this argues that weight is wrong in production.**
- **Frequency is doing real damage.** It spans the full 0–1 at weight 0.2, while semantic spans
  0.10–0.82 at weight 0.5 — a 0.20 swing against a 0.36 one. A frequently-retrieved irrelevant
  memory therefore outranks a highly-relevant one that has never been retrieved.

A third of the problem is upstream of the weights. One-hop graph expansion injects candidates
into the pool with a **hardcoded semantic score of 0.5** (`memory_pipeline.py`), and those were
24.2% of all candidates — while 43% of genuine semantic hits score *below* 0.5. Graph-expanded
memories systematically outrank real matches. Dropping them lifts semantic-only recall@5 from
0.384 to 0.408.

### Re-measured after fixing the bump-timing and graph-expansion bugs

The two bugs above are now fixed in code (frequency bumps only the memories that
survive `top_k` truncation; graph-expanded candidates get a real cosine score
instead of a constant 0.5). Re-running the capture at `--top-k 50` against a
**freshly flushed** frequency store — not the months of accumulated contamination
the numbers above were measured under — gives:

| weights (semantic / temporal / frequency) | recall@5 | recall@10 | recall@20 |
|---|---|---|---|
| 1.0 / 0.0 / 0.0 — semantic only | **0.412** | 0.510 | 0.603 |
| 0.9 / 0.05 / 0.05 | 0.386 | 0.481 | 0.594 |
| 0.8 / 0.1 / 0.1 | 0.337 | 0.458 | 0.563 |
| 0.7 / 0.2 / 0.1 | 0.329 | 0.445 | 0.546 |
| 0.6 / 0.3 / 0.1 | 0.322 | 0.430 | 0.532 |
| 0.5 / 0.3 / 0.2 — as shipped | 0.144 | 0.271 | 0.434 |

The conclusion doesn't change, it sharpens: recall degrades **monotonically** as
frequency weight rises, with a cliff at the shipped 0.2 — every intermediate
point tried (0.05–0.1) recovers most of the gap to semantic-only, so the damage
is disproportionate to the weight, not linear in it. Both code fixes are real
improvements (the bump now means what it claims to, graph expansion is no
longer over-ranked), but neither rescues the frequency term itself, because the
root cause was never the bugs — it's that LoCoMo ingests everything at once and
asks each question once, so no memory has an access history for frequency to
legitimately reflect. `weight_frequency` stays at 0.2 anyway, for the same
reason as before: this is one synthetic benchmark, structurally incapable of
telling frequency's production value from its LoCoMo-specific harm, and that
isn't grounds to silently re-tune a production weight.

### A caveat that applies to every recall number above

Retrieval frequency is a **counter that search itself increments**, so the ranking function
mutates every time it is used. 99.6% of candidates now carry a non-zero frequency score
(median 0.26) accumulated across these eval runs — on a fresh ingest every one would be 0.0.

This makes the benchmark order-dependent: each run re-ranks the next. It is part of why
recall@5 measured 0.233 at `top_k=20` earlier and 0.183 at `top_k=50` here — some of that gap
is the larger pool admitting more graph-injected candidates, and some is simply that the second
measurement ran against a corpus the first one had already re-weighted. **Treat the recall
figures as accurate to roughly ±0.05, not to three decimals**, until the eval resets frequency
counters between runs.

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

- The L2/L3/L4 compression pipeline is implemented, wired to a `celery_beat` service, and fires
  `run_compression_pipeline` on a 86400s schedule; L2 and L3 have tests. What has *not* been
  verified is its behaviour over real elapsed time — every run so far was triggered by hand with
  timestamps forced, so the aging thresholds (`compression_l3_after_days`,
  `compression_l4_after_days`) have never actually been crossed by the clock. L4 additionally
  no-ops unless `S3_ARCHIVE_BUCKET` is set.
- LoCoMo has been run on two conversations out of ten (304 of 1,986 questions). The oracle
  diagnostic now exists and says the answer model, not retrieval, is the ceiling — so the
  remaining eight conversations would sharpen the estimate without changing that conclusion.
- **The frequency term in the score blend halves retrieval quality** (recall@5 0.384 → 0.183),
  and the weights are still `0.5/0.3/0.2` — that number itself is deliberately left untouched,
  because one synthetic benchmark shouldn't silently re-tune production ranking. What *has*
  changed: `redis_client.bump_frequency` used to fire on every candidate the retriever
  returned, before the `top_k` truncation — so memories pushed out by graph expansion and
  never shown to anyone still counted as "retrieved". It now fires only on the memories that
  survive truncation, so the counter means "this was actually returned" instead of "this was
  a candidate." That's a fix to what the signal measures, not to the weight — and it's been
  re-measured (see "Re-measured after fixing the bump-timing and graph-expansion bugs" above):
  the conclusion holds, more starkly than before. `--top-k 20` still compensates for the gap.
- **Graph expansion used to inject candidates at a hardcoded semantic score of 0.5**
  (`memory_pipeline.py`), which outranked 43% of genuine semantic hits. It now batch-retrieves
  each expanded candidate's stored vector from Qdrant and scores it by real cosine similarity
  against the query, falling back to 0.5 only if a vector is unexpectedly missing. Both this and
  the bump-timing fix are baked into the re-measurement above — it's the combined effect of the
  two, not either one in isolation.
- **Prompt injection via stored memory content** — see "Prompt injection via stored memories"
  under Security boundary. Fenced and sanitized, not eliminated; still the top open risk here.
- **The eval is not reproducible run-to-run**: search increments retrieval-frequency counters,
  so the ranking function mutates as the benchmark uses it. Until the harness resets those
  counters per run, repeated measurements drift by roughly ±0.05 recall.
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

MIT — see `LICENSE`. That covers the code in this repository.

It does **not** cover `benchmarks/locomo/locomo10.json`, which is the LoCoMo dataset
redistributed here under **CC BY-NC 4.0** (non-commercial), along with `official_scoring.py`,
adapted from the same source. Citation and attribution are in
[`benchmarks/locomo/README.md`](benchmarks/locomo/README.md). If you vendor this repo into
something commercial, delete `benchmarks/` — nothing under `app/` imports it, so the service
itself stays MIT-clean. The scripts in `scripts/` that evaluate against it go too.
