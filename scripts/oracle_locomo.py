"""Oracle diagnostic for the LoCoMo eval: how well does the *answer model* do
when retrieval is perfect?

eval_locomo.py measures retrieval + generation together. This script removes
retrieval from the loop entirely: instead of searching Engram, it builds the
context from exactly the dialog turns the dataset itself labels as the
supporting evidence for each question (the `evidence` field, e.g. ["D1:2"]).
Everything downstream -- context formatting, the official QA prompt templates,
the model call, the official scorer -- is identical to eval_locomo.py.

So the oracle score is an upper bound on what the end-to-end pipeline can
score with this answer model. If oracle F1 is close to the end-to-end F1, the
answer model (or the scorer's notion of a correct answer) is the bottleneck and
better retrieval cannot help much.

Because LoCoMo's F1 is token-overlap with precision in the denominator, and its
gold answers are ~3-5 words, a verbose-but-correct answer scores badly. To
separate "the model is wrong" from "the model is merely wordy", every score is
reported twice: on the raw completion, and on a trimmed version (first
sentence, parentheticals removed). The trimmed number is a diagnostic only --
it is NOT the official protocol, which scores the raw completion.

Usage:
  python scripts/oracle_locomo.py [--samples conv-30] [--max-tokens 128]

Needs no running services -- only Bedrock credentials. That makes it the
cheapest way to sanity-check the eval before spending a full ingest+search run.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks" / "locomo"))

from official_scoring import CATEGORY_NAMES, score_qa  # noqa: E402

from app.core.bedrock_client import invoke_chat  # noqa: E402

DATA_FILE = Path(__file__).parent.parent / "benchmarks" / "locomo" / "locomo10.json"

# Same templates as eval_locomo.py, verbatim from the paper's task_eval/gpt_utils.py
QA_PROMPT = (
    "\nBased on the above context, write an answer in the form of a short phrase for the "
    'following question. Answer with exact words from the context whenever possible.\n\n'
    "Question: {} Short answer:\n"
)
QA_PROMPT_CAT_5 = (
    "\nBased on the above context, answer the following question.\n\nQuestion: {} Short answer:\n"
)


def trim(text: str) -> str:
    """Diagnostic-only shortening: drop parenthetical asides, keep first sentence."""
    text = re.sub(r"\([^)]*\)?", "", text).strip()
    return re.split(r"(?<=[.!?])\s", text)[0].strip()


def build_turn_index(conv: dict) -> dict[str, str]:
    """dia_id -> context line, formatted exactly as eval_locomo.py ingests turns."""
    index = {}
    for i in range(1, 40):
        key = f"session_{i}"
        if key not in conv:
            continue
        date_time = conv.get(f"session_{i}_date_time", "")
        for turn in conv[key]:
            index[turn["dia_id"]] = f'({date_time}) {turn["speaker"]} said, "{turn["text"]}"'
    return index


def run_sample(sample: dict, max_tokens: int) -> dict:
    sample_id = sample["sample_id"]
    turns = build_turn_index(sample["conversation"])
    results = []

    print(f"[{sample_id}] oracle-answering {len(sample['qa'])} questions "
          f"(max_tokens={max_tokens}, context = gold evidence turns only)...")

    for i, q in enumerate(sample["qa"]):
        evidence = [turns[e] for e in q.get("evidence", []) if e in turns]
        context = "\n".join(f"- {line}" for line in evidence)
        template = QA_PROMPT_CAT_5 if q["category"] == 5 else QA_PROMPT
        prompt = context + "\n" + template.format(q["question"])
        try:
            answer = invoke_chat(prompt, max_tokens=max_tokens).strip()
            f1_raw = score_qa(q, answer)
            f1_trimmed = score_qa(q, trim(answer))
        except Exception as exc:
            answer, f1_raw, f1_trimmed = f"ERROR: {exc}", 0.0, 0.0
        results.append({
            "category": q["category"], "question": q["question"],
            "gold": q.get("answer", q.get("adversarial_answer", "")),
            "prediction": answer, "n_evidence": len(evidence),
            "f1": f1_raw, "f1_trimmed": f1_trimmed,
        })
        if (i + 1) % 20 == 0:
            print(f"[{sample_id}]   {i + 1}/{len(sample['qa'])} done")

    return {"sample_id": sample_id, "max_tokens": max_tokens, "mode": "oracle", "results": results}


def report(all_results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("ORACLE diagnostic -- gold evidence as context (upper bound on end-to-end F1)")
    print("=" * 78)

    by_cat: dict[int, list[tuple[float, float]]] = {}
    for run in all_results:
        for r in run["results"]:
            by_cat.setdefault(r["category"], []).append((r["f1"], r["f1_trimmed"]))

    flat = []
    print(f"  {'category':<14} {'n':<6} {'f1(raw)':<10} {'f1(trimmed)':<12}")
    for cat in sorted(by_cat):
        scores = by_cat[cat]
        flat.extend(scores)
        raw = sum(s[0] for s in scores) / len(scores)
        trm = sum(s[1] for s in scores) / len(scores)
        print(f"  {CATEGORY_NAMES[cat]:<14} {len(scores):<6} {raw:<10.4f} {trm:<12.4f}")

    raw = sum(s[0] for s in flat) / len(flat) if flat else 0.0
    trm = sum(s[1] for s in flat) / len(flat) if flat else 0.0
    print(f"  {'OVERALL':<14} {len(flat):<6} {raw:<10.4f} {trm:<12.4f}")
    print("=" * 78)

    lens = [len(r["prediction"].split()) for run in all_results for r in run["results"]]
    golds = [len(str(r["gold"]).split()) for run in all_results for r in run["results"]]
    print(f"mean prediction length: {sum(lens) / len(lens):.1f} words "
          f"(mean gold length: {sum(golds) / len(golds):.1f} words)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, default="conv-30", help="comma-separated sample_ids")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--out", type=str, default=None, help="write full per-question results as JSON")
    args = parser.parse_args()

    data = json.load(open(DATA_FILE, encoding="utf-8"))
    wanted = set(args.samples.split(","))
    samples = [s for s in data if s["sample_id"] in wanted]
    if not samples:
        sys.exit(f"no samples matched {sorted(wanted)}")

    all_results = [run_sample(s, args.max_tokens) for s in samples]
    report(all_results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
