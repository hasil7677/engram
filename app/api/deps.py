import logging

from fastapi import Header, HTTPException
from psycopg import Error as PsycopgError

from app.core.rate_limit import enforce_rate_limit
from app.core.tenant_store import resolve_tenant
from app.models.schemas import TenantContext

logger = logging.getLogger(__name__)

# An unknown key and a revoked key answer identically on purpose: telling a
# caller "that key used to be valid" confirms a tenant exists to someone
# holding a leaked-then-rotated key. Expiry is distinguished because it's the
# tenant's own live key and the fix ("rotate it") is theirs to make.
_KEY_FAILURES = {
    "unknown": (401, "Invalid API key"),
    "revoked": (401, "Invalid API key"),
    "expired": (401, "API key expired"),
}


def get_tenant_context(
    x_api_key: str = Header(..., alias="X-API-Key"),
    x_user_id: str = Header(..., alias="X-User-Id"),
) -> TenantContext:
    """Two-level tenant model:
      Level 1 (tenant_id) — resolved server-side from the API key, never client-supplied.
      Level 2 (user_id)   — the end-user inside that tenant's app, scoped under tenant_id.

    Every route depends on this. Qdrant/Neo4j queries must filter on BOTH ids.

    SECURITY: X-User-Id is asserted by the caller and NOT authenticated. That is
    correct for a server-side SDK — it's the tenant's own end-user namespace and
    their backend is trusted — but it means the API key must never ship in a
    browser or mobile app. Anyone holding the key can pass any user_id and read
    every end-user's memories in that tenant. Browser clients need short-lived
    per-user tokens minted by the tenant's backend; that doesn't exist yet.
    """
    if not x_user_id:
        raise HTTPException(status_code=400, detail="X-User-Id header required")

    try:
        resolution = resolve_tenant(x_api_key)
    except PsycopgError:
        # Fail closed. Auth sits in front of every route, so a control-plane
        # outage must refuse requests, never wave them through — and it should
        # say "unavailable" rather than "invalid key", or every tenant will
        # spend the outage rotating keys that were fine all along.
        logger.exception("control-plane lookup failed during authentication")
        raise HTTPException(status_code=503, detail="Auth backend unavailable")

    if resolution.reason in _KEY_FAILURES:
        status_code, detail = _KEY_FAILURES[resolution.reason]
        raise HTTPException(status_code=status_code, detail=detail)

    if resolution.reason == "suspended":
        raise HTTPException(
            status_code=403,
            detail=f"Tenant '{resolution.tenant_id}' is suspended",
        )

    if not resolution.ok or resolution.tenant_id is None:
        # Defensive: an unmapped reason must refuse, not fall through to a
        # context with tenant_id=None that later queries would filter on.
        logger.error("unhandled key resolution reason: %s", resolution.reason)
        raise HTTPException(status_code=401, detail="Invalid API key")

    enforce_rate_limit(resolution.tenant_id)
    return TenantContext(tenant_id=resolution.tenant_id, user_id=x_user_id)
