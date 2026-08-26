"""RTFM For Me Agent - documentation assistant built on Redis vector search + RAG.

Package layout (domain-oriented):
    rtfm_agent.api         HTTP layer: FastAPI app, route modules, schemas
    rtfm_agent.ingestion   corpus loading, chunking, indexing, versioning
    rtfm_agent.retrieval   query-time: search, semantic cache, scope, rewrite, RAG
    rtfm_agent.routing     intent classification, actions, memory, cache warming
    rtfm_agent.crawler     web crawl: fetch/safety, job lifecycle, review gate
    rtfm_agent.common      cross-cutting: tenancy, sessions, events, metrics, utils
    rtfm_agent.prompts     versioned prompt templates (.txt) + builders
"""

__version__ = "0.7.1"
