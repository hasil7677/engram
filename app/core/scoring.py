import math
from datetime import datetime, timezone

from app.config import settings


def temporal_decay_score(timestamp: datetime) -> float:
    """Exponential decay: score = e^(-lambda * age_in_days). Recent memories -> ~1.0."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - timestamp).total_seconds() / 86400
    return math.exp(-settings.temporal_decay_lambda * max(age_days, 0))


def normalized_frequency_score(raw_frequency: float, max_frequency: float) -> float:
    """Min-max normalize against the user's max observed frequency so this stays in [0,1]
    and doesn't blow out the weighted sum for power users."""
    if max_frequency <= 0:
        return 0.0
    return min(raw_frequency / max_frequency, 1.0)


def final_score(semantic_score: float, temporal_score: float, frequency_score: float) -> float:
    """Final Score = (semantic * w_sem) + (temporal_decay * w_temp) + (frequency * w_freq).

    Weights are config, not hardcoded — see settings.weight_*. Lets callers retune per
    query type later (e.g. "what did I say yesterday" should weight temporal higher).
    """
    return (
        semantic_score * settings.weight_semantic
        + temporal_score * settings.weight_temporal
        + frequency_score * settings.weight_frequency
    )
