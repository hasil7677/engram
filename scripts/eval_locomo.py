"""Runs Engram against the real, official LoCoMo benchmark
(benchmarks/locomo/locomo10.json, from snap-research/locomo, ACL 2024).

Methodology, matching the paper's own RAG-retriever comparison setup (their
task_eval/rag_utils.py swaps in different retrievers -- DPR/contriever/dragon/
openai -- under a fixed QA prompt + scorer; here Engram is that retriever):
  1. Ingest every dialog turn, across every session, as one Engram memory
     each, via the real POST /v1/memory endpoint (not a direct function call)
     -- same interface a real caller would use. Text is formatted the same
     way the paper's own RAG baseline formats context documents: "(<session
     date>) <speaker> said, "<text>"" -- the session date has to be embedded
     in the text itself, since it's the *conversation's* fictional date, not
     Engram's real ingestion timestamp.
  2. For each of the conversation's real QA questions, retrieve via the real
     POST /v1/memory/search endpoint, then generate a short-phrase answer
     using the *official* QA_PROMPT template (from task_eval/gpt_utils.py) so
     the generation step is the same as the paper's, isolating retrieval
     quality -- via app.core.bedrock_client.invoke_chat directly (same
     Bedrock/Mistral-7B model Engram uses in production, just bypassing
     /v1/chat's own different prompt template for this comparison).
  3. Score with benchmarks/locomo/official_scoring.py, ported verbatim from
     the paper's task_eval/evaluation.py.

Usage: python scripts/eval_locomo.py [--samples conv-30,conv-26] [--top-k 5]
Defaults to the single smallest conversation (conv-30: 369 turns, 105 QA) --
respects Engram's real 60/min per-tenant rate limit, which alone makes a
single conversation take several minutes; the full 10-conversation dataset
(5,882 turns, 1,986 questions) would take roughly 10x longer.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks" / "locomo"))

from official_scoring import CATEGORY_NAMES, score_qa  # noqa: E402

from app.core.bedrock_client import invoke_chat  # noqa: E402

BASE_URL = "http://localhost:8000"
DATA_FILE = Path(__file__).parent.parent / "benchmarks" / "locomo" / "locomo10.json"
COMPOSE_FILE = str(Path(__file__).parent.parent / "docker-compose.yml")

# Official prompts, verbatim from task_eval/gpt_utils.py
QA_PROMPT = (
    "\nBased on the above context, write an answer in the form of a short phrase for the "
    'following question. Answer with exact words from the context whenever possible.\n\n'
    "Question: {} Short answer:\n"
)
QA_PROMPT_CAT_5 = (
    "\nBased on the above context, answer the following question.\n\nQuestion: {} Short answer:\n"
)

import urllib.request  # noqa: E402


def _request(method: str, path: str, api_key: str, user_id: str, body: dict | None = None, timeout: int = 30) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data, method=method,
        headers={"Content-Type": "application/json", "X-API-Key": api_key, "X-User-Id": user_id},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def register_tenant(api_key: str, tenant_id: str) -> None:
    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "exec", "-T", "api", "python", "-c",
         f"from app.core.tenant_store import register_tenant_key; register_tenant_key({api_key!r}, {tenant_id!r})"],
        check=True,
    )


def ingest_conversation(api_key: str, user_id: str, sample: dict, pace_seconds: float) -> int:
    conv = sample["conversation"]
    n_sessions = len([k for k in conv if k.startswith("session_") and not k.endswith("date_time")])
    n = 0
    for i in range(1, n_sessions + 1):
        key = f"session_{i}"
        if key not in conv:
            continue
        date_time = conv.get(f"session_{i}_date_time", "")
        for turn in conv[key]:
            text = f'({date_time}) {turn["speaker"]} said, "{turn["text"]}"'
            _request("POST", "/v1/memory", api_key, user_id, {"text": text})
            n += 1
            time.sleep(pace_seconds)
    return n


def answer_question(api_key: str, user_id: str, question: dict, top_k: int) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = _request("POST", "/v1/memory/search", api_key, user_id,
                       {"query": question["question"], "top_k": top_k, "use_graph_expansion": True})
    retrieval_latency = time.perf_counter() - t0

    # Deliberately not result["context_string"] -- that's built for Engram's own
    # /v1/chat prompt and includes "(relevance=0.70)"-style debug annotations,
    # which leaked into a generated answer in an earlier run (caught by
    # inspecting real predictions, not by inspection of this code). Build a
    # clean context from the raw memories instead, matching the official
    # RAG baseline's plain-document format.
    context = "\n".join(f'- {m["text"]}' for m in result["memories"])
    prompt_template = QA_PROMPT_CAT_5 if question["category"] == 5 else QA_PROMPT
    prompt = context + "\n" + prompt_template.format(question["question"])
    answer = invoke_chat(prompt, max_tokens=32).strip()
    return answer, retrieval_latency


def run_sample(sample: dict, top_k: int, pace_seconds: float, rate_limit_per_min: int) -> dict:
    sample_id = sample["sample_id"]
    tenant_id = f"locomo_{sample_id}"
    api_key = f"locomo_key_{sample_id}"
    user_id = "eval"

    print(f"[{sample_id}] registering tenant + erasing prior state...")
    register_tenant(api_key, tenant_id)
    _request("DELETE", "/v1/memory", api_key, user_id)

    print(f"[{sample_id}] ingesting dialog turns (paced at {pace_seconds}s/req to respect "
          f"the {rate_limit_per_min}/min rate limit)...")
    t0 = time.perf_counter()
    n_turns = ingest_conversation(api_key, user_id, sample, pace_seconds)
    ingest_time = time.perf_counter() - t0
    print(f"[{sample_id}] ingested {n_turns} turns in {ingest_time:.0f}s")

    results = []
    print(f"[{sample_id}] answering {len(sample['qa'])} QA questions...")
    for i, q in enumerate(sample["qa"]):
        try:
            answer, retrieval_latency = answer_question(api_key, user_id, q, top_k)
            f1 = score_qa(q, answer)
        except Exception as exc:
            answer, retrieval_latency, f1 = f"ERROR: {exc}", 0.0, 0.0
        results.append({
            "category": q["category"], "question": q["question"],
            "gold": q.get("answer", q.get("adversarial_answer", "")),
            "prediction": answer, "f1": f1, "retrieval_latency": retrieval_latency,
        })
        if (i + 1) % 20 == 0:
            print(f"[{sample_id}]   {i + 1}/{len(sample['qa'])} done")
        time.sleep(pace_seconds)

    return {"sample_id": sample_id, "n_turns": n_turns, "ingest_time": ingest_time, "results": results}


def report(all_results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("LoCoMo results (official F1 scoring, category-matched)")
    print("=" * 78)

    by_cat: dict[int, list[float]] = {}
    for run in all_results:
        for r in run["results"]:
            by_cat.setdefault(r["category"], []).append(r["f1"])

    all_scores = []
    for cat in sorted(by_cat):
        scores = by_cat[cat]
        all_scores.extend(scores)
        avg = sum(scores) / len(scores)
        print(f"  {CATEGORY_NAMES[cat]:<14} n={len(scores):<5} avg_f1={avg:.4f}")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"  {'OVERALL':<14} n={len(all_scores):<5} avg_f1={overall:.4f}")
    print("=" * 78)

    avg_retrieval_latency = sum(
        r["retrieval_latency"] for run in all_results for r in run["results"]
    ) / sum(len(run["results"]) for run in all_results)
    print(f"avg retrieval latency: {avg_retrieval_latency * 1000:.0f}ms")
    for run in all_results:
        print(f"  {run['sample_id']}: {run['n_turns']} turns ingested in {run['ingest_time']:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, default=None, help="comma-separated sample_ids, default: smallest conversation only")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pace-seconds", type=float, default=1.1, help="delay between rate-limited requests")
    parser.add_argument("--out", type=str, default=None, help="write full per-question results as JSON")
    args = parser.parse_args()

    data = json.load(open(DATA_FILE, encoding="utf-8"))
    if args.samples:
        wanted = set(args.samples.split(","))
        samples = [s for s in data if s["sample_id"] in wanted]
    else:
        samples = [min(data, key=lambda s: sum(
            len(s["conversation"][f"session_{i}"])
            for i in range(1, 40) if f"session_{i}" in s["conversation"]
        ))]

    print(f"Running LoCoMo eval on: {[s['sample_id'] for s in samples]}")

    all_results = [run_sample(s, args.top_k, args.pace_seconds, 60) for s in samples]

    report(all_results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
