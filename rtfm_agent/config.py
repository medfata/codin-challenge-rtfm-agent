"""Central configuration, loaded from environment / .env."""

import os

from dotenv import load_dotenv

load_dotenv()

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
INDEX_NAME = os.getenv("INDEX_NAME", "doc_idx")

# Embeddings (local int8 ONNX export of BAAI/bge-small-en-v1.5)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
ORT_THREADS = int(os.getenv("ORT_THREADS", "2"))

# Docs
DOCS_DIR = os.getenv("DOCS_DIR", "docs/progit2")

# LLM lanes (OpenAI-compatible endpoints).
# Primary lane: Groq fast inference. Fallback lane: Google AI Studio Gemini.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_BASE_URL = (os.getenv("LLM_BASE_URL", "") or "https://api.groq.com/openai/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
# Generation lane model.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
# Fast lane for small classification/rewrite/summary calls (separate quota pool).
LLM_FAST_MODEL = os.getenv("LLM_FAST_MODEL", "openai/gpt-oss-20b")

# Fallback lane: tried automatically on HTTP 429/5xx/connection errors.
ENABLE_LLM_FALLBACK = os.getenv("ENABLE_LLM_FALLBACK", "1") == "1"
FALLBACK_LLM_BASE_URL = (
    os.getenv("FALLBACK_LLM_BASE_URL", "")
    or "https://generativelanguage.googleapis.com/v1beta/openai/"
).rstrip("/")
FALLBACK_LLM_API_KEY = os.getenv("FALLBACK_LLM_API_KEY", "") or GOOGLE_API_KEY
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "gemini-2.5-flash")

# Retrieval tuning
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
# bge-small cosine distances: relevant content lands ~0.15-0.30, unrelated
# noise ~0.45+. Keep this below the noise floor so off-topic questions get
# no context at all (guaranteed refusal instead of hoping the LLM refuses).
RETRIEVAL_MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.40"))

# Semantic cache
CACHE_INDEX_NAME = os.getenv("CACHE_INDEX_NAME", "cache_idx")
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.15"))

# Sessions
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(24 * 3600)))
HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "10"))
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "1") == "1"

# Long-term memory (Redis Agent Memory Server)
MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8002").rstrip("/")
ENABLE_LONG_TERM_MEMORY = os.getenv("ENABLE_LONG_TERM_MEMORY", "1") == "1"
MEMORY_SEARCH_LIMIT = int(os.getenv("MEMORY_SEARCH_LIMIT", "3"))

# Step 6: hybrid search / summarisation / observability
SCOPE_TOP_N_FILES = int(os.getenv("SCOPE_TOP_N_FILES", "3"))
HISTORY_STORE_MAX = int(os.getenv("HISTORY_STORE_MAX", "30"))
RECENT_TURNS_VERBATIM = int(os.getenv("RECENT_TURNS_VERBATIM", "6"))
SUMMARY_CHAR_BUDGET = int(os.getenv("SUMMARY_CHAR_BUDGET", "2000"))
# $ per million tokens (free tier -> 0.0)
PRICE_PER_MTOK_INPUT = float(os.getenv("PRICE_PER_MTOK_INPUT", "0"))
PRICE_PER_MTOK_OUTPUT = float(os.getenv("PRICE_PER_MTOK_OUTPUT", "0"))

# Step 7: semantic routing
ENABLE_ROUTING = os.getenv("ENABLE_ROUTING", "1") == "1"
ENABLE_ACTIONS = os.getenv("ENABLE_ACTIONS", "1") == "1"
# flush_cache / reingest additionally require a keyword match on the raw
# question (see router.DESTRUCTIVE_ACTIONS) before they will execute.
ENABLE_DESTRUCTIVE_ACTIONS = os.getenv("ENABLE_DESTRUCTIVE_ACTIONS", "1") == "1"

# Step 9: document versioning. Auto content-hash tracking per tenant; warns
# when answers come from an older corpus (stale cache) or drifted disk files.
ENABLE_DOC_VERSIONING = os.getenv("ENABLE_DOC_VERSIONING", "1") == "1"
ENABLE_DRIFT_WARNING = os.getenv("ENABLE_DRIFT_WARNING", "1") == "1"
DRIFT_SCAN_TTL_S = int(os.getenv("DRIFT_SCAN_TTL_S", "30"))

# Step 10: MCP exposure. Mounts the assistant as an MCP server at /mcp
# (streamable HTTP) so external AI agents can call it as a tool.
ENABLE_MCP = os.getenv("ENABLE_MCP", "1") == "1"
# Fallback tenant for MCP clients that send no X-Tenant-Id header.
MCP_DEFAULT_TENANT = os.getenv("MCP_DEFAULT_TENANT", "").strip().lower()
# Optional shared-secret check on /mcp (Authorization: Bearer <token>).
MCP_BEARER_TOKEN = os.getenv("MCP_BEARER_TOKEN", "")
# Hostname allowlist for the SDK's DNS-rebinding protection; empty keeps the
# SDK's localhost defaults (set it when deploying behind a real hostname).
MCP_ALLOWED_HOSTS = [
    h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()
]

# Step 11: real-time event notifications. Per-tenant Redis Streams feed the
# SSE endpoint /events/stream so frontends hear about ingestion completions
# and saved conversation turns without polling.
ENABLE_EVENTS = os.getenv("ENABLE_EVENTS", "1") == "1"
# Ring-buffer size per tenant stream; old entries fall off (approximate trim).
EVENTS_STREAM_MAXLEN = int(os.getenv("EVENTS_STREAM_MAXLEN", "1000"))
# Idle window between SSE heartbeat comments; also bounds XREAD blocking.
EVENTS_HEARTBEAT_S = int(os.getenv("EVENTS_HEARTBEAT_S", "15"))

# Step 12: multi-model strategy. Redis doesn't care which model generated the
# content it stores, so background/stored work rides a cheap economy lane and
# only user-facing doc answers use the capable generation lane.
# Economy lane: chitchat replies, memory-route synthesis, semantic-cache
# warming. Defaults ride the Google key; when the Gemini free tier answers
# 429/5xx the lane fails over to a deep-quota Groq model (separate pool from
# the gpt-oss-20b fast lane so warming never starves routing).
LLM_ECONOMY_BASE_URL = (
    os.getenv("LLM_ECONOMY_BASE_URL", "")
    or "https://generativelanguage.googleapis.com/v1beta/openai/"
).rstrip("/")
LLM_ECONOMY_API_KEY = os.getenv("LLM_ECONOMY_API_KEY", "") or GOOGLE_API_KEY
LLM_ECONOMY_MODEL = os.getenv("LLM_ECONOMY_MODEL", "gemini-2.5-flash-lite")
LLM_ECONOMY_FALLBACK_MODEL = os.getenv(
    "LLM_ECONOMY_FALLBACK_MODEL", "llama-3.1-8b-instant"
)
# Cache warming: regenerate popular questions' answers with the economy lane
# straight into the semantic cache (background job, REST + conversational).
ENABLE_CACHE_WARM = os.getenv("ENABLE_CACHE_WARM", "1") == "1"
CACHE_WARM_TOP_N = int(os.getenv("CACHE_WARM_TOP_N", "10"))
# Per-tenant question popularity: ZSET {prefix}qfreq, member=question text,
# score=ask count; bounded and TTL-refreshed on every doc-route question.
QFREQ_TTL_S = int(os.getenv("QFREQ_TTL_S", str(30 * 24 * 3600)))
QFREQ_MAX_ENTRIES = int(os.getenv("QFREQ_MAX_ENTRIES", "200"))
# Browser origins allowed to subscribe cross-origin ("*" for dev; comma list
# to tighten in production; empty disables CORS - same-origin needs none).
EVENTS_CORS_ORIGINS = [
    o.strip() for o in os.getenv("EVENTS_CORS_ORIGINS", "*").split(",") if o.strip()
]

# Step 8: multi-tenancy. Every request (except /health) MUST send X-Tenant-Id.
# Comma-separated allowlist ("acme,globex") or "*" to accept any valid slug.
_TENANTS_RAW = os.getenv("TENANTS", "*").strip()
TENANTS_OPEN = _TENANTS_RAW == "*"
TENANT_ALLOWLIST = frozenset(
    t.strip().lower() for t in _TENANTS_RAW.split(",") if t.strip()
)
