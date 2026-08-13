from fastapi import Header, HTTPException

from app.core.rate_limit import enforce_rate_limit
from app.core.tenant_store import resolve_tenant
from app.models.schemas import TenantContext


def get_tenant_context(
    x_api_key: str = Header(..., alias="X-API-Key"),
    x_user_id: str = Header(..., alias="X-User-Id"),
) -> TenantContext:
    """Two-level tenant model:
      Level 1 (tenant_id) — resolved server-side from the API key, never client-supplied.
      Level 2 (user_id)   — the end-user inside that tenant's app, scoped under tenant_id.

    Every route depends on this. Qdrant/Neo4j queries must filter on BOTH ids.
    """
    tenant_id = resolve_tenant(x_api_key)
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not x_user_id:
        raise HTTPException(status_code=400, detail="X-User-Id header required")
    enforce_rate_limit(tenant_id)
    return TenantContext(tenant_id=tenant_id, user_id=x_user_id)
