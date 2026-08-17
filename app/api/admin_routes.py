from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.admin_deps import require_admin
from app.core.audit import usage_for_tenant
from app.core.tenant_store import (
    create_key,
    list_keys,
    revoke_key,
    revoke_tenant,
    set_tenant_status,
)
from app.db.qdrant_client import count_for_tenant

router = APIRouter(dependencies=[Depends(require_admin)])


class CreateTenantRequest(BaseModel):
    tenant_id: str


class CreateKeyRequest(BaseModel):
    name: str = ""
    expires_in_days: int | None = Field(default=None, gt=0)


class TenantStatusRequest(BaseModel):
    status: str  # active | suspended | deleted


def _key_json(info) -> dict:
    """Metadata only — the plaintext key is returned exactly once, by the
    endpoint that mints it, and is unrecoverable afterwards."""
    return {
        "id": info.id,
        "key_prefix": info.key_prefix,
        "name": info.name,
        "created_at": info.created_at.isoformat(),
        "last_used_at": info.last_used_at.isoformat() if info.last_used_at else None,
        "expires_at": info.expires_at.isoformat() if info.expires_at else None,
        "revoked_at": info.revoked_at.isoformat() if info.revoked_at else None,
        "active": info.active,
    }


@router.post("/tenants")
def create_tenant(body: CreateTenantRequest):
    """Self-service tenant onboarding: generates a fresh API key server-side —
    never accepts a caller-supplied one. Shown once; only its hash is stored."""
    api_key, info = create_key(body.tenant_id, name="initial")
    return {"tenant_id": body.tenant_id, "api_key": api_key, "key": _key_json(info)}


@router.post("/tenants/{tenant_id}/keys")
def issue_key(tenant_id: str, body: CreateKeyRequest):
    """Additional key for an existing tenant, leaving current keys working.

    This is half of zero-downtime rotation: issue here, deploy the new key,
    then DELETE the old key by id. Revoking the tenant wholesale (the DELETE
    /tenants/{id} route) is the incident path, not the rotation path.
    """
    api_key, info = create_key(
        tenant_id, name=body.name, expires_in_days=body.expires_in_days
    )
    return {"tenant_id": tenant_id, "api_key": api_key, "key": _key_json(info)}


@router.get("/tenants/{tenant_id}/keys")
def get_keys(tenant_id: str):
    return {"tenant_id": tenant_id, "keys": [_key_json(k) for k in list_keys(tenant_id)]}


@router.delete("/tenants/{tenant_id}/keys/{key_id}")
def delete_key(tenant_id: str, key_id: int):
    """Revokes one key. Scoped to the tenant in the path so a mistyped key_id
    can't retire some other tenant's key."""
    owned = {k.id for k in list_keys(tenant_id)}
    if key_id not in owned:
        raise HTTPException(status_code=404, detail="No such key for this tenant")
    if not revoke_key(key_id):
        raise HTTPException(status_code=409, detail="Key already revoked")
    return {"tenant_id": tenant_id, "key_id": key_id, "revoked": True}


@router.put("/tenants/{tenant_id}/status")
def update_tenant_status(tenant_id: str, body: TenantStatusRequest):
    """Suspend or reactivate without touching keys — so a tenant who lapses on
    payment and then pays resumes on their existing integration rather than
    having to re-key it."""
    try:
        updated = set_tenant_status(tenant_id, body.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not updated:
        raise HTTPException(status_code=404, detail="No such tenant")
    return {"tenant_id": tenant_id, "status": body.status}


@router.delete("/tenants/{tenant_id}")
def deactivate_tenant(tenant_id: str):
    """Revokes every API key registered for this tenant — does not delete the
    tenant's stored memories (see the per-user DELETE /v1/memory for that)."""
    revoked = revoke_tenant(tenant_id)
    return {"tenant_id": tenant_id, "revoked_keys": revoked}


@router.get("/tenants/{tenant_id}/usage")
def tenant_usage(tenant_id: str):
    return {
        "tenant_id": tenant_id,
        "total_memories": count_for_tenant(tenant_id),
        "actions": usage_for_tenant(tenant_id),
    }
