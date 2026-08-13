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

See `../../scripts/eval_locomo.py` for how Engram is run against this data.
