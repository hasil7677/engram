"""Retrieval-only evaluation: does Engram actually return the right memories?

The LoCoMo F1 in eval_locomo.py is an end-to-end score -- retrieval and answer
generation collapsed into one number -- and oracle_locomo.py showed the answer
model dominates it. So F1 says very little about Engram specifically.

This script measures retrieval on its own. Every LoCoMo question ships the
dialog turns that support it (the `evidence` field, e.g. ["D1:2"]). Those turns
were ingested verbatim, one memory each, so a retrieved memory can be mapped
back to its dia_id by exact text match and compared against that gold set. No
answer model is involved and nothing is scored by F1.

Two metrics, both standard for retrieval:
  hit@k    -- fraction of questions where at least one gold evidence turn is in
              the top k. "Did we surface anything useful at all?"
  recall@k -- fraction of a question's gold evidence turns that are in the top
              k, averaged over questions. Stricter, and the one that matters for
              multi-hop questions whose evidence spans several turns.

Requires the conversation to already be ingested (eval_locomo.py does that).
This does searches only -- no ingest, no Bedrock calls -- so it is cheap to
re-run while tuning retrieval.

Usage: python scripts/recall_locomo.py [--samples conv-30,conv-26] [--top-k 20]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks" / "locomo"))

from official_scoring import CATEGORY_NAMES  # noqa: E402

from eval_locomo import _request  # noqa: E402

DATA_FILE = Path(__file__).parent.parent / "benchmarks" / "locomo" / "locomo10.json"

# Report the curve at these cutoffs. One search per question at the largest k,
# then slice -- the ranking is stable, so there's no reason to search repeatedly.
K_VALUES = [1, 5, 10, 20]


def build_text_to_dia_id(conv: dict) -> dict[str, str]:
    """Reverse of the ingest formatting in eval_locomo.ingest_conversation.

    The exact string Engram stores is the join key back to a dia_id, which is
    what the dataset's `evidence` field refers to.
    """
    index = {}
    for i in range(1, 40):
        key = f"session_{i}"
        if key not in conv:
            continue
        date_time = conv.get(f"session_{i}_date_time", "")
        for turn in conv[key]:
            text = f'({date_time}) {turn["speaker"]} said, "{turn["text"]}"'
            index[text] = turn["dia_id"]
    return index


def run_sample(sample: dict, top_k: int, pace_seconds: float, use_graph_expansion: bool = True) -> dict:
    sample_id = sample["sample_id"]
    api_key = f"locomo_key_{sample_id}"
    user_id = "eval"
    text_to_dia = build_text_to_dia_id(sample["conversation"])

    print(f"[{sample_id}] searching {len(sample['qa'])} questions (top_k={top_k}, "
          f"graph_expansion={use_graph_expansion}, {len(text_to_dia)} turns ingested)...")

    results = []
    unmatched_total = 0
    for i, q in enumerate(sample["qa"]):
        gold = [e for e in q.get("evidence", []) if e]
        resp = _request("POST", "/v1/memory/search", api_key, user_id,
                        {"query": q["question"], "top_k": top_k,
                         "use_graph_expansion": use_graph_expansion})
        retrieved, unmatched, components = [], 0, []
        for m in resp["memories"]:
            dia_id = text_to_dia.get(m["text"])
            if dia_id is None:
                unmatched += 1
            retrieved.append(dia_id)
            # Keep the three components, not just final_score. The blend weights are
            # applied server-side, so recording the parts is what makes an offline
            # weight sweep possible -- otherwise every candidate weighting costs
            # another full search pass against the API.
            components.append({
                "semantic": m["semantic_score"], "temporal": m["temporal_score"],
                "frequency": m["frequency_score"], "final": m["final_score"],
            })
        unmatched_total += unmatched

        results.append({
            "category": q["category"], "question": q["question"],
            "gold_evidence": gold, "retrieved": retrieved, "components": components,
            "n_returned": len(resp["memories"]), "n_unmatched": unmatched,
        })
        if (i + 1) % 40 == 0:
            print(f"[{sample_id}]   {i + 1}/{len(sample['qa'])} done")
        time.sleep(pace_seconds)

    if unmatched_total:
        # Every returned memory should map back to a turn. If not, the ingest
        # format and this script's reverse map have drifted apart, and the
        # numbers below would silently understate recall.
        print(f"[{sample_id}] WARNING: {unmatched_total} retrieved memories did not map "
              f"to a dia_id -- recall is understated; check the ingest format")

    return {"sample_id": sample_id, "top_k": top_k,
            "use_graph_expansion": use_graph_expansion, "results": results}


def sweep(path: str) -> None:
    """Re-rank a saved run under different blend weights, without re-searching.

    The API applies the blend server-side, but each returned memory carries its
    three components, and the blend is just a weighted sum -- so one search pass
    makes every weighting testable offline. This is the feedback loop for tuning
    settings.weight_*: capture once at a large --top-k, then sweep for free.

    Exact for candidates already in the pool. A weighting different enough to
    have pulled in a memory the captured run never returned is not represented,
    so treat large deltas as directional and confirm with a fresh search.
    """
    runs = json.load(open(path, encoding="utf-8"))
    rows = [r for run in runs for r in run["results"] if r.get("components")]
    if not rows:
        sys.exit(f"{path} has no component scores -- re-run recall_locomo.py to capture them")

    configs = [(1.0, 0.0, 0.0), (0.9, 0.05, 0.05), (0.8, 0.1, 0.1),
               (0.7, 0.2, 0.1), (0.6, 0.3, 0.1), (0.5, 0.3, 0.2)]

    print("=" * 78)
    print(f"Blend sweep over {len(rows)} questions (pool={len(rows[0]['components'])} per question)")
    print("=" * 78)
    print(f"  {'semantic/temporal/frequency':<30} {'recall@5':<11} {'recall@10':<11} {'recall@20':<11}")
    for ws, wt, wf in configs:
        cells = []
        for k in (5, 10, 20):
            scores = []
            for r in rows:
                gold = set(r["gold_evidence"])
                if not gold:
                    continue
                ranked = sorted(
                    zip(r["components"], r["retrieved"]),
                    key=lambda p: -(p[0]["semantic"] * ws + p[0]["temporal"] * wt
                                    + p[0]["frequency"] * wf),
                )
                top = {d for _, d in ranked[:k] if d is not None}
                scores.append(len(gold & top) / len(gold))
            cells.append(sum(scores) / len(scores))
        label = f"{ws}/{wt}/{wf}" + ("   (as shipped)" if (ws, wt, wf) == (0.5, 0.3, 0.2) else "")
        print(f"  {label:<30} {cells[0]:<11.4f} {cells[1]:<11.4f} {cells[2]:<11.4f}")
    print("=" * 78)


def _metrics(rows: list[dict], k: int) -> tuple[float, float]:
    """(hit@k, recall@k) over rows that actually have gold evidence."""
    hits, recalls = [], []
    for r in rows:
        gold = set(r["gold_evidence"])
        if not gold:
            continue
        top = set(x for x in r["retrieved"][:k] if x is not None)
        found = len(gold & top)
        hits.append(1.0 if found else 0.0)
        recalls.append(found / len(gold))
    if not hits:
        return 0.0, 0.0
    return sum(hits) / len(hits), sum(recalls) / len(recalls)


def report(all_results: list[dict], top_k: int) -> None:
    rows = [r for run in all_results for r in run["results"]]
    ks = [k for k in K_VALUES if k <= top_k]

    print("\n" + "=" * 78)
    print("Retrieval-only evaluation -- gold evidence turns from the LoCoMo dataset")
    print("=" * 78)
    print(f"  {'cutoff':<10} {'hit@k':<12} {'recall@k':<12}")
    for k in ks:
        hit, rec = _metrics(rows, k)
        print(f"  {'k=' + str(k):<10} {hit:<12.4f} {rec:<12.4f}")

    print(f"\n  by category (at k={ks[-1]}):")
    print(f"  {'category':<14} {'n':<6} {'hit@k':<12} {'recall@k':<12}")
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        hit, rec = _metrics(sub, ks[-1])
        print(f"  {CATEGORY_NAMES[cat]:<14} {len(sub):<6} {hit:<12.4f} {rec:<12.4f}")

    n_gold = sum(len(r["gold_evidence"]) for r in rows)
    print(f"\n  {len(rows)} questions, {n_gold} gold evidence turns "
          f"({n_gold / len(rows):.1f} per question)")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, default="conv-30,conv-26")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--pace-seconds", type=float, default=1.1)
    parser.add_argument("--no-graph-expansion", action="store_true",
                        help="search without one-hop graph expansion, to isolate its effect "
                             "on ranking (expanded memories enter the pool with a hardcoded "
                             "semantic score)")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--sweep", type=str, default=None, metavar="RESULTS_JSON",
                        help="skip searching; re-rank a saved results file under a range of "
                             "blend weights and print recall@k for each")
    args = parser.parse_args()

    if args.sweep:
        sweep(args.sweep)
        return

    data = json.load(open(DATA_FILE, encoding="utf-8"))
    wanted = set(args.samples.split(","))
    samples = [s for s in data if s["sample_id"] in wanted]
    if not samples:
        sys.exit(f"no samples matched {sorted(wanted)}")

    all_results = [run_sample(s, args.top_k, args.pace_seconds, not args.no_graph_expansion)
                   for s in samples]
    report(all_results, args.top_k)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
