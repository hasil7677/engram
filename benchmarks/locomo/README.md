# LoCoMo benchmark data

`locomo10.json` is the official dataset from the LoCoMo benchmark:

> Maharana, A., Lee, D-H., Tulyakov, S., Bansal, M., Barbieri, F., Fang, Y.
> "Evaluating Very Long-Term Conversational Memory of LLM Agents." ACL 2024.
> https://arxiv.org/abs/2402.17753
> Source: https://github.com/snap-research/locomo

Licensed **CC BY-NC 4.0** (non-commercial). Used here for benchmark evaluation only.

`official_scoring.py` is adapted from the repo's `task_eval/evaluation.py` (same license) —
the F1/exact-match scoring functions only, ported to drop the unused `bert_score`/ROUGE paths
so this doesn't need `bert_score`/`rouge` as dependencies.

Three scripts run against this data: `../../scripts/eval_locomo.py` (end-to-end F1),
`../../scripts/oracle_locomo.py` (oracle diagnostic — bypasses retrieval, needs no services),
and `../../scripts/recall_locomo.py` (retrieval-only hit@k/recall@k, no answer model).

## Result files

Each holds the full per-question record — question, gold answer, raw prediction, F1 — so a
score can be re-derived or re-scored without re-running anything. The suffix is the answer
model's token cap, which turned out to matter enough to be worth naming:

| File | What it is |
|---|---|
| `conv-30_results_maxtok32.json` | The original run. Mean F1 0.149, but 48 of 105 predictions were cut off mid-sentence. |
| `conv-30_results_maxtok128.json` | Same conversation, truncation fixed. Mean F1 0.133 — *lower*, because the scorer penalises the extra words. |
| `conv-26_results_maxtok128.json` | Second conversation, 199 questions. Mean F1 0.126. |
| `conv-30_oracle.json`, `conv-26_oracle.json` | Oracle runs: context is the dataset's own `evidence` turns instead of retrieval. Combined F1 0.191, which is the ceiling the answer model imposes. |
| `recall_conv-30_conv-26.json` | Retrieval-only run — per question, the gold evidence turn ids and the ranked turn ids Engram actually returned. No answer model involved, so hit@k/recall@k can be recomputed at any cutoff without re-searching. |
| `results_topk20.json` | Both conversations at `--top-k 20` instead of 5, same memories, no re-ingest. Mean F1 0.176 — the headline number, +37% relative over top_k=5. |

The oracle files carry an extra `f1_trimmed` field per question — the same prediction rescored
after dropping parentheticals and keeping only the first sentence. It is a diagnostic for
separating wrong answers from merely wordy ones, and is **not** part of the official protocol.

See the Benchmark section of the root `README.md` for what these numbers mean.
