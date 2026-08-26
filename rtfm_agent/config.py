"""Central configuration, loaded from environment / .env.

Settings are grouped into domain sections on a single ``settings`` object.
Every section is an independent pydantic-settings model reading its own
environment variables; every variable name is preserved exactly as before
via AliasChoices, so existing .env files and deployments keep working
unchanged:

    settings.redis.url            <- REDIS_URL
    settings.llm.fast_model       <- LLM_FAST_MODEL
    settings.crawl.max_pages      <- CRAWL_MAX_PAGES
"""

from functools import lru_cache
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _split_csv(value) -> list[str]:
    """Accept a real list (JSON/.env object) or a comma-separated string."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


class _Section(BaseSettings):
    """Base for all sections: read .env, ignore unrelated variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Allow constructor overrides by field name alongside env aliases.
        populate_by_name=True,
    )


class RedisSettings(_Section):
    url: str = Field(default="redis://localhost:6379", validation_alias=AliasChoices("REDIS_URL"))
    index_name: str = Field(default="doc_idx", validation_alias=AliasChoices("INDEX_NAME"))


class EmbeddingSettings(_Section):
    # Local int8 ONNX export of BAAI/bge-small-en-v1.5
    model: str = Field(default="BAAI/bge-small-en-v1.5", validation_alias=AliasChoices("EMBEDDING_MODEL"))
    dim: int = Field(default=384, validation_alias=AliasChoices("EMBEDDING_DIM"))
    ort_threads: int = Field(default=2, validation_alias=AliasChoices("ORT_THREADS"))


class DocsSettings(_Section):
    dir: str = Field(default="docs/progit2", validation_alias=AliasChoices("DOCS_DIR"))
    web_dir: str = Field(default="docs/web", validation_alias=AliasChoices("WEB_DOCS_DIR"))


class LlmSettings(_Section):
    # Google AI Studio key: backs the fallback + default economy lanes and is
    # surfaced in /health reporting.
    google_api_key: str = Field(default="", validation_alias=AliasChoices("GOOGLE_API_KEY"))

    # Generation lane (OpenAI-compatible endpoints). Primary lane: Groq fast
    # inference. Fallback lane: Google AI Studio Gemini.
    base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        validation_alias=AliasChoices("LLM_BASE_URL"),
    )
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "GROQ_API_KEY"),
    )
    model: str = Field(default="openai/gpt-oss-120b", validation_alias=AliasChoices("LLM_MODEL"))
    # Fast lane for small classification/rewrite/summary calls (separate quota pool).
    fast_model: str = Field(default="openai/gpt-oss-20b", validation_alias=AliasChoices("LLM_FAST_MODEL"))

    # Fallback lane: tried automatically on HTTP 429/5xx/connection errors.
    enable_fallback: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_LLM_FALLBACK"))
    fallback_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        validation_alias=AliasChoices("FALLBACK_LLM_BASE_URL"),
    )
    fallback_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("FALLBACK_LLM_API_KEY"),
    )
    fallback_model: str = Field(default="gemini-2.5-flash", validation_alias=AliasChoices("FALLBACK_LLM_MODEL"))

    # Economy lane: chitchat replies, memory-route synthesis, semantic-cache
    # warming. Defaults ride the Google key; when the Gemini free tier answers
    # 429/5xx the lane fails over to a deep-quota Groq model - a pool separate
    # from the gpt-oss-20b fast lane so warming never starves routing.
    economy_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        validation_alias=AliasChoices("LLM_ECONOMY_BASE_URL"),
    )
    economy_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_ECONOMY_API_KEY"),
    )
    economy_model: str = Field(default="gemini-3.5-flash-lite", validation_alias=AliasChoices("LLM_ECONOMY_MODEL"))
    economy_fallback_model: str = Field(
        default="qwen/qwen3.6-27b",
        validation_alias=AliasChoices("LLM_ECONOMY_FALLBACK_MODEL"),
    )

    @model_validator(mode="after")
    def _normalize(self) -> "LlmSettings":
        self.base_url = self.base_url.rstrip("/")
        self.fallback_base_url = self.fallback_base_url.rstrip("/")
        self.economy_base_url = self.economy_base_url.rstrip("/")
        # GOOGLE_API_KEY is the shared last-resort credential for both the
        # fallback and economy lanes (matches the original env-var defaults).
        self.fallback_api_key = self.fallback_api_key or self.google_api_key
        self.economy_api_key = (
            self.economy_api_key or self.google_api_key or self.fallback_api_key
        )
        return self


class RetrievalSettings(_Section):
    k: int = Field(default=5, validation_alias=AliasChoices("RETRIEVAL_K"))
    # bge-small cosine distances: relevant content lands ~0.15-0.30, unrelated
    # noise ~0.45+. Below the noise floor so off-topic questions get no context
    # at all (guaranteed refusal instead of hoping the LLM refuses).
    max_distance: float = Field(default=0.40, validation_alias=AliasChoices("RETRIEVAL_MAX_DISTANCE"))
    # Max files a scope hint may narrow the search to (hybrid search filter).
    scope_top_n_files: int = Field(default=3, validation_alias=AliasChoices("SCOPE_TOP_N_FILES"))


class CacheSettings(_Section):
    index_name: str = Field(default="cache_idx", validation_alias=AliasChoices("CACHE_INDEX_NAME"))
    threshold: float = Field(default=0.15, validation_alias=AliasChoices("CACHE_THRESHOLD"))


class SessionSettings(_Section):
    ttl_seconds: int = Field(default=24 * 3600, validation_alias=AliasChoices("SESSION_TTL_SECONDS"))
    history_max_messages: int = Field(default=10, validation_alias=AliasChoices("HISTORY_MAX_MESSAGES"))
    enable_query_rewrite: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_QUERY_REWRITE"))
    store_max: int = Field(default=30, validation_alias=AliasChoices("HISTORY_STORE_MAX"))
    recent_turns_verbatim: int = Field(default=6, validation_alias=AliasChoices("RECENT_TURNS_VERBATIM"))
    summary_char_budget: int = Field(default=2000, validation_alias=AliasChoices("SUMMARY_CHAR_BUDGET"))


class MemorySettings(_Section):
    server_url: str = Field(
        default="http://localhost:8002",
        validation_alias=AliasChoices("MEMORY_SERVER_URL"),
    )
    enabled: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_LONG_TERM_MEMORY"))
    search_limit: int = Field(default=3, validation_alias=AliasChoices("MEMORY_SEARCH_LIMIT"))

    @model_validator(mode="after")
    def _strip_url(self) -> "MemorySettings":
        self.server_url = self.server_url.rstrip("/")
        return self


class PricingSettings(_Section):
    # $ per million tokens (free tier -> 0.0)
    input_per_mtok: float = Field(default=0.0, validation_alias=AliasChoices("PRICE_PER_MTOK_INPUT"))
    output_per_mtok: float = Field(default=0.0, validation_alias=AliasChoices("PRICE_PER_MTOK_OUTPUT"))


class RoutingSettings(_Section):
    enabled: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_ROUTING"))
    actions_enabled: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_ACTIONS"))
    # flush_cache / reingest additionally require a keyword match on the raw
    # question (see routing.intent.DESTRUCTIVE_ACTIONS) before they execute.
    destructive_enabled: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_DESTRUCTIVE_ACTIONS"))


class VersioningSettings(_Section):
    # Auto content-hash tracking per tenant; warns when answers come from an
    # older corpus (stale cache) or drifted disk files.
    enabled: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_DOC_VERSIONING"))
    drift_warnings: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_DRIFT_WARNING"))
    drift_scan_ttl_s: int = Field(default=30, validation_alias=AliasChoices("DRIFT_SCAN_TTL_S"))


class McpSettings(_Section):
    # Mounts the assistant as an MCP server so external AI agents can call it.
    enabled: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_MCP"))
    # Fallback tenant for MCP clients that send no X-Tenant-Id header.
    default_tenant: str = Field(default="", validation_alias=AliasChoices("MCP_DEFAULT_TENANT"))
    # Optional shared-secret check on /mcp (Authorization: Bearer <token>).
    bearer_token: str = Field(default="", validation_alias=AliasChoices("MCP_BEARER_TOKEN"))
    # Hostname allowlist for the SDK's DNS-rebinding protection; empty keeps
    # the SDK's localhost defaults (set it when deploying behind a hostname).
    allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        validation_alias=AliasChoices("MCP_ALLOWED_HOSTS"),
    )

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, value):
        return _split_csv(value)

    @model_validator(mode="after")
    def _normalize(self) -> "McpSettings":
        self.default_tenant = self.default_tenant.strip().lower()
        return self


class EventsSettings(_Section):
    # Per-tenant Redis Streams feed the SSE endpoint /events/stream so
    # frontends hear about ingestions and saved turns without polling.
    enabled: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_EVENTS"))
    # Ring-buffer size per tenant stream; old entries fall off (exact trim).
    stream_maxlen: int = Field(default=1000, validation_alias=AliasChoices("EVENTS_STREAM_MAXLEN"))
    # Idle window between SSE heartbeat comments; also bounds XREAD blocking.
    heartbeat_s: int = Field(default=15, validation_alias=AliasChoices("EVENTS_HEARTBEAT_S"))
    # Browser origins allowed to subscribe cross-origin ("*" for dev; comma
    # list to tighten in production; empty disables CORS - same-origin needs none).
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"],
        validation_alias=AliasChoices("EVENTS_CORS_ORIGINS"),
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        return _split_csv(value)


class WarmSettings(_Section):
    # Cache warming: regenerate popular questions' answers with the economy
    # lane straight into the semantic cache (background job, REST + chat).
    enabled: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_CACHE_WARM"))
    top_n: int = Field(default=10, validation_alias=AliasChoices("CACHE_WARM_TOP_N"))
    # Per-tenant question popularity: ZSET {prefix}qfreq; bounded + TTL-refreshed.
    qfreq_ttl_s: int = Field(default=30 * 24 * 3600, validation_alias=AliasChoices("QFREQ_TTL_S"))
    qfreq_max_entries: int = Field(default=200, validation_alias=AliasChoices("QFREQ_MAX_ENTRIES"))


class TenantsSettings(_Section):
    # Every request (except /health) MUST send X-Tenant-Id. Comma-separated
    # allowlist ("acme,globex") or "*" to accept any valid slug.
    raw: str = Field(default="*", validation_alias=AliasChoices("TENANTS"))

    @property
    def open(self) -> bool:
        return self.raw.strip() == "*"

    @property
    def allowlist(self) -> frozenset[str]:
        return frozenset(t.strip().lower() for t in self.raw.split(",") if t.strip())


class CrawlSettings(_Section):
    # POST /crawl discovers documentation on a website (sitemap first, then
    # same-host link BFS), extracts readable text, and STAGES it for human
    # review; approval merges pages into docs/web/<org>/ and triggers ingestion.
    enabled: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_WEB_CRAWL"))
    # Runaway guards: page count / link depth caps (a request may lower them).
    max_pages: int = Field(default=50, validation_alias=AliasChoices("CRAWL_MAX_PAGES"))
    max_depth: int = Field(default=3, validation_alias=AliasChoices("CRAWL_MAX_DEPTH"))
    # Hard request cap per crawl regardless of requested max_pages.
    hard_page_cap: int = Field(default=500, validation_alias=AliasChoices("CRAWL_HARD_PAGE_CAP"))
    # Politeness: sleep between fetches; per-request timeout.
    delay_ms: int = Field(default=750, validation_alias=AliasChoices("CRAWL_DELAY_MS"))
    timeout_s: float = Field(default=15.0, validation_alias=AliasChoices("CRAWL_TIMEOUT_S"))
    # Response body cap (bytes) and minimum extracted text to keep a page.
    max_bytes: int = Field(default=2 * 1024 * 1024, validation_alias=AliasChoices("CRAWL_MAX_BYTES"))
    min_text_chars: int = Field(default=200, validation_alias=AliasChoices("CRAWL_MIN_TEXT_CHARS"))
    # SSRF guard rejects private/loopback/link-local targets; the override
    # exists for tests crawling a localhost fixture - never enable in prod.
    allow_private_hosts: bool = Field(default=False, validation_alias=AliasChoices("CRAWL_ALLOW_PRIVATE_HOSTS"))
    # Staged jobs awaiting review expire after this many hours (swept lazily).
    stage_ttl_h: float = Field(default=24.0, validation_alias=AliasChoices("CRAWL_STAGE_TTL_H"))


class Settings:
    """Aggregate root; access everything as ``settings.<section>.<field>``.

    A plain container (not a BaseSettings model) so section names never
    collide with environment-variable lookups.
    """

    def __init__(self) -> None:
        self.redis = RedisSettings()
        self.embed = EmbeddingSettings()
        self.docs = DocsSettings()
        self.llm = LlmSettings()
        self.retrieval = RetrievalSettings()
        self.cache = CacheSettings()
        self.sessions = SessionSettings()
        self.memory = MemorySettings()
        self.pricing = PricingSettings()
        self.routing = RoutingSettings()
        self.versioning = VersioningSettings()
        self.mcp = McpSettings()
        self.events = EventsSettings()
        self.warm = WarmSettings()
        self.tenants = TenantsSettings()
        self.crawl = CrawlSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
