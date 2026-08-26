"""Pydantic request/response models for every REST endpoint."""

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    docs_dir: str | None = Field(default=None, max_length=500)


class IngestResponse(BaseModel):
    documents: int
    chunks_generated: int
    chunks_stored: int
    stale_keys_removed: int
    index_created: bool
    index: str
    embedding_dim: int
    duration_s: float
    tenant: str
    docs_dir: str
    corpus_version: int = 0
    digest: str = ""
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    session_id: str | None = Field(default=None, max_length=100)


class Citation(BaseModel):
    source_file: str
    section_heading: str
    chunk_pos: int
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources_consulted: int
    cached: bool = False
    session_id: str
    tenant: str = ""
    documents_scoped: list[str] = []
    route: str = "doc"
    action: str | None = None
    stale: bool = False
    warning: str | None = None


class MetricsResponse(BaseModel):
    requests_total: int
    errors_total: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_cached_ms: float
    avg_uncached_ms: float
    total_questions: int
    prompt_tokens_total: int
    completion_tokens_total: int
    estimated_cost_usd: float
    cache_size: int
    route_doc_total: int = 0
    route_chitchat_total: int = 0
    route_action_total: int = 0
    route_memory_total: int = 0
    stale_answers_served: int = 0
    mcp_calls_total: int = 0
    events_published_total: int = 0
    llm_calls_generation_total: int = 0
    llm_calls_fast_total: int = 0
    llm_calls_economy_total: int = 0
    cache_warm_runs_total: int = 0
    cache_warm_answers_total: int = 0
    crawl_jobs_total: int = 0
    crawl_pages_fetched_total: int = 0
    crawl_failures_total: int = 0
    crawl_discarded_total: int = 0


class CrawlRequest(BaseModel):
    start_url: str = Field(min_length=9, max_length=2000)
    max_pages: int | None = Field(default=None, ge=1, le=500)
    max_depth: int | None = Field(default=None, ge=0, le=10)
    path_prefix: str | None = Field(default=None, max_length=500)
    # Trusted-source escape hatch: skip the review gate entirely.
    auto_ingest: bool = False


class CrawlApproveRequest(BaseModel):
    exclude: list[str] = Field(default_factory=list, max_length=1000)
