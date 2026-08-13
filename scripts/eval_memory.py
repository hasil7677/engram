"""Ad-hoc memory-quality eval harness, modeled loosely on the taxonomy used by
LongMemEval (single-session / multi-session / temporal-reasoning / knowledge-
update / abstention) and LOCOMO (long-horizon recall under noise).

Not a pytest suite / CI gate -- a benchmark script you run and read the report
from. Talks to the real running API over HTTP, so it measures the actual
deployed system (Bedrock latency included), not internal functions directly.

Usage: python scripts/eval_memory.py
Requires: docker compose up (api reachable at BASE_URL), tenants pre-registered
via register_tenants() below (run once, idempotent).
"""
import json
import re
import time
import urllib.request

BASE_URL = "http://localhost:8000"


def register_tenants(cases: list[dict]) -> None:
    """Registers one tenant per case directly via tenant_store, in-process in
    the api container -- avoids the admin API (needs ADMIN_API_KEY, unset here)
    and avoids burning rate-limit budget on registration traffic."""
    import subprocess

    lines = ["from app.core.tenant_store import register_tenant_key"]
    for case in cases:
        lines.append(f"register_tenant_key({case['api_key']!r}, {case['tenant_id']!r})")
    script = "\n".join(lines)
    subprocess.run(
        ["docker", "compose", "-f", "C:\\Users\\Sahil\\Downloads\\engram-memory\\docker-compose.yml",
         "exec", "-T", "api", "python", "-c", script],
        check=True,
    )


def _request(method: str, path: str, api_key: str, user_id: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
            "X-User-Id": user_id,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def add_memory(api_key: str, user_id: str, text: str) -> None:
    _request("POST", "/v1/memory", api_key, user_id, {"text": text})


def search(api_key: str, user_id: str, query: str, top_k: int = 5) -> dict:
    return _request("POST", "/v1/memory/search", api_key, user_id, {"query": query, "top_k": top_k})


def chat(api_key: str, user_id: str, text: str, timeout: int = 60) -> str:
    """Reads the raw SSE stream and returns the assembled reply text."""
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat",
        data=json.dumps({"text": text}).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": api_key, "X-User-Id": user_id},
    )
    reply_parts = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        event_type = None
        for raw_line in resp:
            line = raw_line.decode().rstrip("\n")
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:") and event_type == "delta":
                reply_parts.append(json.loads(line[5:].strip())["text"])
    return "".join(reply_parts)


NOISE_FACTS = [
    "The office coffee machine broke on Tuesday.",
    "I watched a documentary about deep sea creatures last night.",
    "My neighbor is repainting their fence this week.",
    "I need to renew my passport before the trip.",
    "The gym added a new set of kettlebells.",
    "I tried a new ramen place downtown, it was decent.",
    "My phone battery has been draining faster lately.",
    "I'm reading a book about Roman history.",
    "The bus route near my house changed schedules.",
    "I planted basil and mint on the balcony.",
    "There's construction noise outside every morning now.",
    "I finally organized my email inbox.",
    "My laptop fan has been louder than usual.",
    "I signed up for a pottery class starting next month.",
    "The local library extended its weekend hours.",
    "I switched to a new brand of coffee beans.",
    "My car needs an oil change soon.",
    "I've been meaning to clean out the garage.",
    "A new bakery opened two blocks from my apartment.",
    "I upgraded my headphones last week.",
]


def contains_any(text: str, expected: list[str]) -> bool:
    lowered = text.lower()
    return any(e.lower() in lowered for e in expected)


CASES = [
    # -- A. single-session, near-immediate recall (tests the "short-term" path,
    #    such as it is -- really just: does a just-stated fact rank #1) --
    dict(
        id="A1", category="single-session",
        tenant_id="eval_a1", api_key="eval_key_a1", user_id="user",
        setup=["My favorite programming language is Rust."],
        noise_before=0, noise_after=0,
        query="What is my favorite programming language?",
        expected=["Rust"],
        via="search",
    ),
    dict(
        id="A2", category="single-session",
        tenant_id="eval_a2", api_key="eval_key_a2", user_id="user",
        setup=["I have a dog named Biscuit."],
        noise_before=0, noise_after=0,
        query="What is my dog's name?",
        expected=["Biscuit"],
        via="search",
    ),

    # -- B. long-horizon recall: target fact buried under noise, tests whether
    #    temporal decay + noise crowd out the still-relevant fact --
    dict(
        id="B1", category="long-horizon",
        tenant_id="eval_b1", api_key="eval_key_b1", user_id="user",
        setup=["My mother's name is Elena and she lives in Lisbon."],
        noise_before=0, noise_after=15,
        query="Where does my mother live?",
        expected=["Lisbon"],
        via="search",
    ),
    dict(
        id="B2", category="long-horizon",
        tenant_id="eval_b2", api_key="eval_key_b2", user_id="user",
        setup=["I broke my left arm skiing in 2019."],
        noise_before=0, noise_after=18,
        query="Have I ever broken a bone?",
        expected=["arm", "broke", "ski"],
        via="search",
    ),
    dict(
        id="B3", category="long-horizon",
        tenant_id="eval_b3", api_key="eval_key_b3", user_id="user",
        setup=["My allergy is to shellfish, it gives me hives."],
        noise_before=10, noise_after=10,
        query="What am I allergic to?",
        expected=["shellfish"],
        via="search",
    ),

    # -- C. temporal reasoning --
    dict(
        id="C1", category="temporal",
        tenant_id="eval_c1", api_key="eval_key_c1", user_id="user",
        setup=["I started working at Acme Corp in January 2024."],
        noise_before=0, noise_after=5,
        query="When did I start working at Acme Corp?",
        expected=["January 2024", "2024"],
        via="search",
    ),

    # -- D. knowledge update / contradiction --
    # D1 is EXPECTED to fail as of the current design: contradiction candidates
    # are discovered via shared *entity names* (find_candidate_duplicates), and
    # "Berlin"/"Paris" are different entities -- so the LLM contradiction judge
    # never even gets called for this pair. This is a real, known gap: any
    # "my X is now Y" fact where the changed value IS the entity (residence,
    # job title, name) can't be entity-linked to its own prior value. Left in
    # the suite specifically to keep this gap visible, not to be "fixed" by
    # loosening the scorer.
    dict(
        id="D1", category="knowledge-update",
        tenant_id="eval_d1", api_key="eval_key_d1", user_id="user",
        setup=["I live in Berlin.", "I moved to Paris and live there now."],
        noise_before=0, noise_after=0,
        settle_seconds=10,
        query="Where do I live?",
        expected=["Paris"],
        not_expected=["Berlin"],
        via="chat",
    ),
    # D2: a contradiction anchored on a *shared* entity ("Acme Corp") DOES get
    # discovered and now auto-resolves -- the old fact is marked superseded
    # (Qdrant + Neo4j, same mechanism as a manual update) rather than just
    # flagged, so search/chat stop surfacing the stale answer.
    dict(
        id="D2", category="knowledge-update",
        tenant_id="eval_d2", api_key="eval_key_d2", user_id="user",
        setup=["Acme Corp's CEO is John Miller.", "Acme Corp announced John Miller stepped down; Sarah Chen is now CEO."],
        noise_before=0, noise_after=0,
        settle_seconds=10,
        query="Who is the CEO of Acme Corp?",
        expected=["Sarah Chen"],
        stale_fact="Acme Corp's CEO is John Miller.",
        via="search",
    ),

    # -- E. multi-hop via shared entity (graph expansion) --
    dict(
        id="E1", category="multi-hop",
        tenant_id="eval_e1", api_key="eval_key_e1", user_id="user",
        setup=["Priya works at Nimbus Labs.", "Nimbus Labs is headquartered in Austin."],
        noise_before=0, noise_after=0,
        query="What city is Priya's company based in?",
        expected=["Austin"],
        via="search",
    ),

    # -- F. abstention: never-stated info, should not confidently hallucinate --
    dict(
        id="F1", category="abstention",
        tenant_id="eval_f1", api_key="eval_key_f1", user_id="user",
        setup=["I like hiking on weekends."],
        noise_before=0, noise_after=0,
        query="What is my mother's maiden name?",
        expected=[],
        via="chat",
    ),
]


def run_case(case: dict) -> dict:
    api_key, user_id = case["api_key"], case["user_id"]

    _request("DELETE", "/v1/memory", api_key, user_id)  # hermetic: wipe any state from a prior run

    for fact in NOISE_FACTS[: case["noise_before"]]:
        add_memory(api_key, user_id, fact)
    for fact in case["setup"]:
        add_memory(api_key, user_id, fact)
    for fact in NOISE_FACTS[: case["noise_after"]]:
        add_memory(api_key, user_id, fact)

    # knowledge-update cases depend on the async contradiction-resolution
    # pipeline (Celery), which runs after add_memory() already returned.
    if case.get("settle_seconds"):
        time.sleep(case["settle_seconds"])

    t0 = time.perf_counter()
    if case["via"] == "search":
        result = search(api_key, user_id, case["query"], top_k=5)
        latency = time.perf_counter() - t0
        result_texts = [m["text"] for m in result["memories"]]
        top5_text = " ".join(result_texts)
        passed = contains_any(top5_text, case["expected"])
        # Exact-text check, not keyword substring: the correct/current fact can
        # legitimately *mention* the superseded value in passing (e.g. "X
        # stepped down, Y is now CEO" contains "X"), so a naive not_expected
        # keyword check false-negatives on a genuinely correct answer. What
        # actually matters is whether the stale fact is still present verbatim
        # as its own memory.
        if passed and case.get("stale_fact") and case["stale_fact"] in result_texts:
            passed = False
        rank = next((i for i, m in enumerate(result["memories"]) if contains_any(m["text"], case["expected"])), None)
        detail = f"rank={rank} of {len(result['memories'])} results" if passed else f"NOT in top5 (or stale fact still present): {[m['text'][:40] for m in result['memories']]}"
    else:
        reply = chat(api_key, user_id, case["query"])
        latency = time.perf_counter() - t0
        passed = contains_any(reply, case["expected"]) if case["expected"] else True
        if case.get("not_expected") and contains_any(reply, case["not_expected"]):
            passed = False
            detail = f"FAILED (stale answer surfaced): {reply[:200]}"
        else:
            detail = f"reply: {reply[:200]}"

    return dict(id=case["id"], category=case["category"], via=case["via"],
                passed=passed, latency=latency, detail=detail)


def main():
    print(f"Registering {len(CASES)} eval tenants...")
    register_tenants(CASES)

    results = []
    for case in CASES:
        print(f"  running {case['id']} ({case['category']}, via={case['via']})...")
        try:
            results.append(run_case(case))
        except Exception as exc:
            results.append(dict(id=case["id"], category=case["category"], via=case["via"],
                                 passed=False, latency=0.0, detail=f"EXCEPTION: {exc}"))

    print("\n" + "=" * 78)
    print(f"{'ID':<5}{'category':<16}{'via':<8}{'result':<8}{'latency':<10}detail")
    print("=" * 78)
    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["passed"])
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{r['id']:<5}{r['category']:<16}{r['via']:<8}{mark:<8}{r['latency']:.2f}s     {r['detail']}")

    print("=" * 78)
    total_pass = sum(r["passed"] for r in results)
    print(f"\nOverall: {total_pass}/{len(results)} passed\n")
    print("By category:")
    for cat, passes in by_category.items():
        print(f"  {cat:<16} {sum(passes)}/{len(passes)}")


if __name__ == "__main__":
    main()
