"""LoCoMo's official F1/exact-match scoring, ported verbatim (minus the unused
bert_score/ROUGE paths) from snap-research/locomo's task_eval/evaluation.py,
so scores here are comparable to the paper's published numbers.

Category numbers match the dataset: 1=multi-hop, 2=temporal, 3=open-domain
knowledge, 4=single-hop, 5=adversarial.
"""
import regex
import string
from collections import Counter

import numpy as np
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def normalize_answer(s: str) -> str:
    s = s.replace(",", "")

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi_hop(prediction: str, ground_truth: str) -> float:
    """Category 1: split both sides on commas into sub-answers, take the best
    prediction-sub-answer match for each ground-truth sub-answer, average."""
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return float(np.mean([max(f1_score(p, gt) for p in predictions) for gt in ground_truths]))


def score_qa(question: dict, prediction: str) -> float:
    """One QA item's score, using the same category-dependent rule as the
    official eval_question_answering(): category 1 uses split multi-hop F1,
    2/3/4 use plain F1, 5 (adversarial, no ground-truth answer) scores 1.0 if
    the prediction correctly abstains, else 0.0.
    """
    category = question["category"]
    if category == 5:
        low = prediction.lower()
        return 1.0 if ("no information available" in low or "not mentioned" in low) else 0.0

    answer = question["answer"]
    if category == 3:
        answer = answer.split(";")[0].strip()

    if category == 1:
        return f1_multi_hop(prediction, answer)
    return f1_score(prediction, answer)


CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
