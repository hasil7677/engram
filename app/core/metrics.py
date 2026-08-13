from prometheus_client import Counter, Histogram

# Deliberately no tenant_id label on any of these — Prometheus label values
# create a new time series each, so a per-tenant label here would give every
# new signup its own unbounded set of series. Per-tenant usage already lives
# in the audit log (app.core.audit); these are aggregate operational metrics.

memory_writes_total = Counter("engram_memory_writes_total", "Memories written")
memory_updates_total = Counter("engram_memory_updates_total", "Memories superseded via update")
memory_deletes_total = Counter("engram_memory_deletes_total", "Memories deleted")
memory_searches_total = Counter("engram_memory_searches_total", "Searches executed")
semantic_cache_hits_total = Counter("engram_semantic_cache_hits_total", "Semantic cache hits")

search_latency_seconds = Histogram("engram_search_latency_seconds", "search_memory latency in seconds")

bedrock_calls_total = Counter("engram_bedrock_calls_total", "Bedrock invoke_chat calls")
bedrock_call_errors_total = Counter("engram_bedrock_call_errors_total", "Bedrock invoke_chat errors")
bedrock_call_latency_seconds = Histogram("engram_bedrock_call_latency_seconds", "Bedrock invoke_chat latency in seconds")

rate_limit_rejections_total = Counter(
    "engram_rate_limit_rejections_total", "Requests rejected by the per-tenant rate limiter"
)

http_requests_total = Counter(
    "engram_http_requests_total", "HTTP requests", ["method", "path_template", "status_code"]
)
http_request_duration_seconds = Histogram(
    "engram_http_request_duration_seconds", "HTTP request duration in seconds", ["method", "path_template"]
)
