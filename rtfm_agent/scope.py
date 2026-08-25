"""Scope detection: map natural document references to indexed source files.

Two layers:
  1. Heuristic regex on the raw question ("based on the getting started guide...")
  2. The rewrite LLM may emit a "SOURCE: <filename>" line for follow-ups

Hints are fuzzy-matched against the actual indexed inventory (distinct
source_file TAG values from Redis), producing an OR-set of files to filter on.
"""

import logging
import re
import time

from redis import Redis

from rtfm_agent.config import SCOPE_TOP_N_FILES
from rtfm_agent.tenancy import TenantContext

logger = logging.getLogger(__name__)

_HINT_RE = re.compile(
    r"\b(?:based on|from|in|according to|per)\s+(?:the\s+)?"
    r"[\"']?(.{3,60}?)[\"']?\s+"
    r"(?:guide|docs|documentation|reference|manual|chapter|section|file|book)",
    re.IGNORECASE,
)

_inventory_cache: dict[str, tuple[float, list[str]]] = {}
_INVENTORY_TTL_S = 300


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _normalize(obj):
    """Recursively decode redis-py's bytes-keyed FT.AGGREGATE response."""
    if isinstance(obj, bytes):
        return obj.decode(errors="replace")
    if isinstance(obj, dict):
        return {_normalize(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    return obj


def get_inventory(r: Redis, t: TenantContext) -> list[str]:
    """Distinct source_file TAG values indexed for this tenant (cached 5 min).

    Cache entries are keyed by tenant id, so each tenant gets its own view.
    """
    global _inventory_cache
    now = time.time()
    cached = _inventory_cache.get(t.id)
    if cached is not None and now - cached[0] < _INVENTORY_TTL_S:
        return cached[1]
    try:
        res = r.execute_command(
            "FT.AGGREGATE", t.doc_index, "*",
            "LOAD", "1", "@source_file",
            "GROUPBY", "1", "@source_file",
            "REDUCE", "COUNT", "0", "AS", "n",
        )
        data = _normalize(res).get("results") or []
        files = []
        for row in data:
            fields = row.get("extra_attributes") or row.get("fields") or {}
            if fields.get("source_file"):
                files.append(fields["source_file"])
        _inventory_cache[t.id] = (now, files)
        return files
    except Exception as exc:
        logger.warning("scope inventory lookup failed: %s", exc)
        return []


def match_files(hint: str, inventory: list[str], top_n: int = SCOPE_TOP_N_FILES) -> list[str]:
    """Fuzzy-match a hint against file paths via token overlap."""
    hint_tokens = _tokenize(hint)
    if not hint_tokens:
        return []
    scored = []
    for path in inventory:
        path_tokens = _tokenize(path.replace(".asc", ""))
        overlap = len(hint_tokens & path_tokens)
        # Reward coverage of the hint by the path, penalise very long paths mildly
        score = overlap / max(len(hint_tokens), 1) - len(path_tokens) * 0.001
        if overlap > 0:
            scored.append((score, path))
    scored.sort(reverse=True)
    return [p for _, p in scored[:top_n]]


def detect_doc_hint(question: str) -> str | None:
    """Layer 1: regex extraction of a document reference from the raw question."""
    m = _HINT_RE.search(question)
    if m:
        hint = m.group(1).strip()
        if hint and not hint.lower().startswith(("this ", "that ", "it")):
            return hint
    return None


def resolve_scope(r: Redis, t: TenantContext, question: str,
                  llm_hint: str | None = None) -> set[str] | None:
    """Return an OR-set of source files to filter on, or None when unscoped.

    Matches hints against the tenant's own indexed inventory only.
    """
    hint = detect_doc_hint(question) or llm_hint
    if not hint:
        return None
    inventory = get_inventory(r, t)
    if not inventory:
        return None
    files = match_files(hint, inventory)
    if files:
        logger.info("scoped search: hint=%r -> %s", hint, files)
        return set(files)
    logger.info("scoped search: hint=%r matched nothing; ignoring", hint)
    return None

