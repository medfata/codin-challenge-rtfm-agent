"""Semantic routing: one fused LLM intent call per message.

Classifies every incoming message into one of four routes before any RAG
work happens:
  doc      - a documentation question (full RAG pipeline; default)
  chitchat - small talk (persona reply, no retrieval)
  action   - operational request (deterministic handler)
  memory   - "what do you know about me?" (recall + synthesis)

The same call also produces the standalone rewritten search query and an
optional source-file hint for doc-route questions, replacing the previous
SOURCE:-line convention.

Fail-open everywhere: malformed JSON, LLM errors, or unknown values degrade
to the doc route with the original question untouched.
"""

import json
import logging
import re
from dataclasses import dataclass

from rtfm_agent import llm as llm_client
from rtfm_agent.config import ENABLE_ROUTING, LLM_FAST_MODEL
from rtfm_agent.prompts import ROUTE_SYSTEM, build_route_prompt

logger = logging.getLogger(__name__)

ROUTES = ("doc", "chitchat", "action", "memory")
ACTIONS = ("metrics", "list_docs", "clear_session", "flush_cache", "reingest")

# Actions with side effects require corroboration in the raw question text -
# the LLM hint alone never triggers them.
DESTRUCTIVE_ACTIONS = frozenset({"flush_cache", "reingest"})

_ACTION_GUARDS = {
    "flush_cache": re.compile(
        r"(?:\b(?:flush|clear|wipe|purge|empty|reset)\b[^.?!]{0,40}\bcache\b"
        r"|\bcache\b[^.?!]{0,40}\b(?:flush|clear|wipe|purge|empty|reset)\b)",
        re.IGNORECASE,
    ),
    "reingest": re.compile(
        r"(?:\b(?:re[\s-]?index|re[\s-]?ingest|re[\s-]?load|ingest)\b[^.?!]{0,50}"
        r"\b(?:docs?|documents?|documentation|corpus|book|files?)\b"
        r"|\b(?:docs?|documents?|documentation|corpus|book)\b[^.?!]{0,50}"
        r"\b(?:re[\s-]?index|re[\s-]?ingest|re[\s-]?load|ingest)\b)",
        re.IGNORECASE,
    ),
}


@dataclass
class RouteResult:
    """Outcome of the fused intent call."""

    route: str = "doc"
    action: str | None = None
    query: str = ""
    source_hint: str | None = None


def classify(question: str, hist_turns: list[dict]) -> RouteResult:
    """One fused intent call: route + action + standalone query + source hint.

    Fail-open: any failure returns the doc route with the original question.
    """
    if not ENABLE_ROUTING:
        return RouteResult(query=question)
    try:
        raw, _ = _intent_call(question, hist_turns)
    except Exception as exc:
        logger.warning("intent call failed (non-fatal): %s", exc)
        return RouteResult(query=question)

    result = _parse(raw, original=question)
    logger.info(
        "intent: route=%s action=%s source=%s",
        result.route, result.action, result.source_hint,
    )
    return result


def _intent_call(question: str, hist_turns: list[dict]) -> tuple[str, dict | None]:
    """Call the LLM in JSON mode; retry without response_format on HTTP 400."""
    messages = [
        {"role": "system", "content": ROUTE_SYSTEM},
        {"role": "user", "content": build_route_prompt(hist_turns, question)},
    ]
    try:
        # Generous cap: gpt-oss models emit hidden reasoning tokens first.
        return llm_client.chat(
            messages,
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
            model=LLM_FAST_MODEL,
        )
    except llm_client.LLMError as exc:
        if "400" not in str(exc):
            raise
        logger.info("response_format unsupported; retrying intent call without it")
        return llm_client.chat(messages, temperature=0.0, max_tokens=256,
                               model=LLM_FAST_MODEL)


def _parse(raw: str, original: str) -> RouteResult:
    """Defensively extract the intent JSON; any doubt degrades to doc."""
    fallback = RouteResult(query=original)
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return fallback
        obj = json.loads(raw[start:end + 1])
        if not isinstance(obj, dict):
            return fallback
    except (json.JSONDecodeError, TypeError, ValueError):
        return fallback

    route = str(obj.get("route") or "").strip().lower()
    if route not in ROUTES:
        return fallback

    action = obj.get("action")
    if isinstance(action, str):
        action = action.strip().lower() or None
    else:
        action = None
    if action is not None and action not in ACTIONS:
        action = None

    source = obj.get("source")
    if isinstance(source, str):
        source = source.strip().strip('"') or None
    else:
        source = None

    query = str(obj.get("query") or "").strip().strip('"')

    if route == "action":
        if action is None:
            return fallback  # action route without a valid verb -> doc
        if action in DESTRUCTIVE_ACTIONS and not _ACTION_GUARDS[action].search(original):
            logger.info("destructive action '%s' rejected by keyword guard", action)
            return fallback  # side effects need corroboration in the raw text
        return RouteResult(route="action", action=action, query=original)

    if route != "doc":
        return RouteResult(route=route, query=original)

    # doc route: adopt the rewritten query only when it looks usable.
    lowered = query.lower()
    if not query or len(query) < 8 or lowered.startswith(("user", "assistant")):
        query = original
    return RouteResult(route="doc", query=query, source_hint=source)
