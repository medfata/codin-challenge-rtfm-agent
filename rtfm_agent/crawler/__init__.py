"""Web crawl domain: URL-safe fetching, background crawl jobs, and the
human review gate that stages pages before they enter the corpus."""

from rtfm_agent.crawler.fetch import (
    CrawlError,
    SeedURL,
    extract_content,
    normalize_url,
    page_id_for,
)
from rtfm_agent.crawler.job import (
    JobNotFoundError,
    JobStateError,
    start_job,
    staging_root,
    sweep_expired,
    tenant_web_root,
)
from rtfm_agent.crawler.review import (
    approve_job,
    discard_job,
    get_job,
    list_jobs,
    mark_ingested,
    read_page,
)

__all__ = [
    "CrawlError",
    "JobNotFoundError",
    "JobStateError",
    "SeedURL",
    "approve_job",
    "discard_job",
    "extract_content",
    "get_job",
    "list_jobs",
    "mark_ingested",
    "normalize_url",
    "page_id_for",
    "read_page",
    "staging_root",
    "start_job",
    "sweep_expired",
    "tenant_web_root",
]
