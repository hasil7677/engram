from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.admin_deps import require_admin
from app.core.tenant_store import generate_and_register_tenant, revoke_tenant
from app.db.qdrant_client import count_for_tenant

router = APIRouter(dependencies=[Depends(require_admin)])


class CreateTenantRequest(BaseModel):
    tenant_id: str


@router.post("/tenants")
def create_tenant(body: CreateTenantRequest):
    """Self-service tenant onboarding: generates a fresh API key server-side —
    never accepts a caller-supplied one. Shown once; only its hash is stored."""
    api_key = generate_and_register_tenant(body.tenant_id)
    return {"tenant_id": body.tenant_id, "api_key": api_key}


@router.delete("/tenants/{tenant_id}")
def deactivate_tenant(tenant_id: str):
    """Revokes every API key registered for this tenant — does not delete the
    tenant's stored memories (see the per-user DELETE /v1/memory for that)."""
    revoked = revoke_tenant(tenant_id)
    return {"tenant_id": tenant_id, "revoked_keys": revoked}


@router.get("/tenants/{tenant_id}/usage")
def tenant_usage(tenant_id: str):
    return {"tenant_id": tenant_id, "total_memories": count_for_tenant(tenant_id)}
