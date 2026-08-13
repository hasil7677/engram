import secrets

from fastapi import Header, HTTPException

from app.config import settings


def require_admin(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> None:
    """Separate from the tenant auth model entirely — this API creates and
    revokes tenants, so it's gated by a single operator secret (ADMIN_API_KEY),
    not a tenant's own API key. Fails closed: an unset admin key means the
    admin API is unusable, never that it's open.
    """
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Admin API not configured")
    if not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")
