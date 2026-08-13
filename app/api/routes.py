from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import get_tenant_context
from app.core import memory_pipeline
from app.core.audit import export_user_data
from app.core.chat import stream_chat_reply
from app.models.schemas import ChatIn, MemoryIn, SearchQuery, SearchResult, TenantContext

router = APIRouter()


@router.post("/memory")
def add_memory(memory_in: MemoryIn, ctx: TenantContext = Depends(get_tenant_context)):
    memory_id = memory_pipeline.add_memory(ctx.tenant_id, ctx.user_id, memory_in)
    return {"memory_id": memory_id}


@router.post("/chat")
def chat(chat_in: ChatIn, ctx: TenantContext = Depends(get_tenant_context)):
    """Recall -> LLM reply, streamed -> persist both turns. See app/core/chat.py."""
    return StreamingResponse(
        stream_chat_reply(ctx.tenant_id, ctx.user_id, chat_in),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/memory/search", response_model=SearchResult)
def search_memory(query: SearchQuery, ctx: TenantContext = Depends(get_tenant_context)):
    return memory_pipeline.search_memory(
        ctx.tenant_id, ctx.user_id, query.query, query.top_k, query.use_graph_expansion
    )


@router.get("/memory/export")
def export_memory(ctx: TenantContext = Depends(get_tenant_context)):
    """GDPR data-portability: full audit trail of everything held on this user."""
    return {"records": export_user_data(ctx.tenant_id, ctx.user_id)}


@router.put("/memory/{memory_id}")
def update_memory(memory_id: str, memory_in: MemoryIn, ctx: TenantContext = Depends(get_tenant_context)):
    """Supersedes memory_id with a new version — the old one is kept as history."""
    new_id = memory_pipeline.update_memory(ctx.tenant_id, ctx.user_id, memory_id, memory_in)
    if new_id is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"memory_id": new_id, "supersedes": memory_id}


@router.get("/memory/{memory_id}/history")
def get_memory_history(memory_id: str, ctx: TenantContext = Depends(get_tenant_context)):
    history = memory_pipeline.get_memory_history(ctx.tenant_id, ctx.user_id, memory_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"history": history}


@router.delete("/memory/{memory_id}")
def delete_memory(memory_id: str, ctx: TenantContext = Depends(get_tenant_context)):
    deleted = memory_pipeline.delete_memory(ctx.tenant_id, ctx.user_id, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True}


@router.delete("/memory")
def erase_all_memory(ctx: TenantContext = Depends(get_tenant_context)):
    """GDPR right to erasure: permanently deletes everything held on this user."""
    return memory_pipeline.erase_user(ctx.tenant_id, ctx.user_id)
