from collections.abc import Iterator

import boto3

from app.config import settings
from app.core.metrics import bedrock_call_errors_total, bedrock_call_latency_seconds, bedrock_calls_total

_client = None


def get_bedrock_client():
    global _client
    if _client is None:
        kwargs = {"region_name": settings.aws_region}
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        _client = boto3.client("bedrock-runtime", **kwargs)
    return _client


def invoke_chat(prompt: str, max_tokens: int = 512, model_id: str | None = None) -> str:
    """One-shot chat completion via Bedrock's model-agnostic Converse API —
    works the same way regardless of which model provider BEDROCK_MODEL_ID points at.

    model_id overrides BEDROCK_MODEL_ID for a single call. Production never passes
    it; the LoCoMo oracle uses it to answer the same questions with a different
    model over an otherwise identical code path, which is what makes the
    answer-model comparison meaningful rather than a rewrite.
    """
    client = get_bedrock_client()
    bedrock_calls_total.inc()
    try:
        with bedrock_call_latency_seconds.time():
            response = client.converse(
                modelId=model_id or settings.bedrock_model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
            )
    except Exception:
        bedrock_call_errors_total.inc()
        raise
    return response["output"]["message"]["content"][0]["text"].strip()


def invoke_chat_stream(prompt: str, max_tokens: int = 512) -> Iterator[str]:
    """Streaming counterpart to invoke_chat, via Bedrock's Converse API stream
    variant. Yields text deltas as they arrive so the caller can forward them
    to a client without waiting for the full completion.
    """
    client = get_bedrock_client()
    bedrock_calls_total.inc()
    try:
        with bedrock_call_latency_seconds.time():
            response = client.converse_stream(
                modelId=settings.bedrock_model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
            )
            for event in response["stream"]:
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                if "text" in delta:
                    yield delta["text"]
    except Exception:
        bedrock_call_errors_total.inc()
        raise
