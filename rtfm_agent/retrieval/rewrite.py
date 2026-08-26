"""Follow-up query rewriting and rolling-conversation summarisation.

Both helpers ride the fast LLM lane; failures are non-fatal and fall back
to the original text.
"""

import logging

from rtfm_agent import llm as llm_client
from rtfm_agent.common.metrics import record_lane_call
from rtfm_agent.config import settings
from rtfm_agent.llm import LLMError
from rtfm_agent.prompts import (
    REWRITE_SYSTEM,
    SUMMARY_SYSTEM,
    build_rewrite_prompt,
    build_summary_prompt,
)

logger = logging.getLogger(__name__)


def rewrite_query(question: str, hist_turns: list[dict],
                  r=None, t=None) -> tuple[str, str | None]:
    """Follow-up handling: standalone query + optional SOURCE document hint.

    Pass `r`/`t` to record the fast-lane metric call; omitted by CLI/tests.
    """
    if not (hist_turns and settings.sessions.enable_query_rewrite):
        return question, None
    try:
        raw, _ = llm_client.chat(
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": build_rewrite_prompt(hist_turns, question)},
            ],
            temperature=0.0,
            max_tokens=512,
            model=settings.llm.fast_model,
        )
        if r is not None and t is not None:
            record_lane_call(r, t, "fast")
        raw = raw.strip()
    except LLMError as exc:
        logger.warning("query rewrite failed (non-fatal): %s", exc)
        return question, None

    source_hint = None
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) >= 2 and lines[-1].upper().startswith("SOURCE:"):
        candidate = lines[-1][len("SOURCE:"):].strip().strip('"')
        if candidate:
            source_hint = candidate
        lines = lines[:-1]

    query = " ".join(lines).strip().strip('"')
    if (
        not query
        or len(query) < 8
        or query.lower().startswith(("user", "assistant"))
    ):
        return question, source_hint
    return query, source_hint


def summarize_turns(existing_summary: str | None, older_msgs,
                    r=None, t=None) -> str | None:
    """Fold older turns (+ previous summary) into a compact rolling summary."""
    raw, _ = llm_client.chat(
        [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": build_summary_prompt(existing_summary, older_msgs)},
        ],
        temperature=0.0,
        max_tokens=512,
        model=settings.llm.fast_model,
    )
    if r is not None and t is not None:
        record_lane_call(r, t, "fast")
    return raw.strip() or None
