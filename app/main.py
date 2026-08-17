import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.api.admin_routes import router as admin_router
from app.api.routes import router
from app.core.metrics import http_request_duration_seconds, http_requests_total
from app.db.neo4j_client import ensure_constraints
from app.db.postgres import init_schema
from app.db.qdrant_client import ensure_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Control plane first: if the tenants/api_keys tables aren't there, every
    # authenticated route 503s anyway, so there's no point warming the memory
    # engines behind a broken front door.
    init_schema()
    ensure_collection()
    ensure_constraints()
    yield


app = FastAPI(title="Engram Memory Middleware", lifespan=lifespan)

# Dev-only chat UI runs on Vite's default ports. Auth here is a header (X-API-Key),
# never a cookie, so a permissive origin list carries no CSRF risk the way
# cookie-based auth would -- a third-party page still can't read the response
# without the browser allowing it, and can't forge the header itself.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/v1")
app.include_router(admin_router, prefix="/v1/admin")


@app.middleware("http")
async def track_http_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    # use the matched route's path template ("/v1/memory/{memory_id}"), never the
    # raw URL, so per-memory ids don't blow up the metric into unbounded series
    route = request.scope.get("route")
    path_template = route.path if route else request.url.path

    http_requests_total.labels(
        method=request.method, path_template=path_template, status_code=response.status_code
    ).inc()
    http_request_duration_seconds.labels(method=request.method, path_template=path_template).observe(duration)
    return response


@app.get("/metrics")
def metrics():
    """Unauthenticated by design, like a standard Prometheus exporter — carries
    no tenant/user data, only aggregate counters. Firewall this off from the
    public internet in a real deployment; it's an ops endpoint, not a tenant API.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"}
