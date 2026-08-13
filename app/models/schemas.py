from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TenantContext(BaseModel):
    """Resolved by the auth middleware on every request. Never trust client-supplied
    tenant_id/user_id directly — these come from the API key lookup."""

    tenant_id: str
    user_id: str


class MemoryIn(BaseModel):
    text: str
    role: str = "user"  # user | assistant | system
    metadata: dict = Field(default_factory=dict)


class ChatIn(BaseModel):
    text: str


class MemoryRecord(BaseModel):
    memory_id: str
    tenant_id: str
    user_id: str
    text: str
    role: str
    timestamp: datetime
    metadata: dict = Field(default_factory=dict)


class SearchQuery(BaseModel):
    query: str
    top_k: int = 10
    use_graph_expansion: bool = True


class ScoredMemory(BaseModel):
    memory_id: str
    text: str
    semantic_score: float
    temporal_score: float
    frequency_score: float
    final_score: float
    timestamp: datetime
    source: str  # "qdrant" | "graph_expansion"


class SearchResult(BaseModel):
    context_string: str
    memories: list[ScoredMemory]
