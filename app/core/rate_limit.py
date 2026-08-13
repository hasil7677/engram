import time

from fastapi import HTTPException

from app.config import settings
from app.core.metrics import rate_limit_rejections_total
from app.db.redis_client import get_redis


def enforce_rate_limit(tenant_id: str) -> None:
    """Fixed-window rate limit keyed on tenant (the API key/developer), not the
    end-user — this protects the tenant's own quota and the shared Bedrock/LLM
    spend sitting behind it. Raises 429 once the tenant exceeds the per-minute
    budget for the current window.
    """
    r = get_redis()
    window = int(time.time() // 60)
    key = f"ratelimit:{tenant_id}:{window}"
    count = r.incr(key)
    if count == 1:
        r.expire(key, 60)
    if count > settings.rate_limit_per_minute:
        rate_limit_rejections_total.inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded, try again shortly")
