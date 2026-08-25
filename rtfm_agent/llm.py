"""Minimal OpenAI-compatible chat client with ordered provider fallback.

Lanes (Step 12): every call names a lane and gets its own provider chain.
  generation - Groq capable model -> Gemini fallback (user-facing answers)
  fast       - Groq small model on the primary endpoint -> Gemini fallback
  economy    - cheap model for stored/background content (chitchat, memory
               synthesis, cache warming): Gemini flash-lite -> deep-quota
               Groq fallback. Redis doesn't care which model generated the
               content it stores; only live doc answers need the big pool.

Lane 1 (primary): Groq fast inference. Lane 2 (optional fallback):
Google AI Studio Gemini, used when a lane's first hop rate-limits (429),
returns 5xx, or is unreachable. Hard 4xx responses are raised immediately -
a bad request must not silently burn the next lane.
"""

import json
import logging
from typing import Iterator

import httpx

from rtfm_agent.config import (
    ENABLE_LLM_FALLBACK,
    FALLBACK_LLM_API_KEY,
    FALLBACK_LLM_BASE_URL,
    FALLBACK_LLM_MODEL,
    GOOGLE_API_KEY,  # re-exported for health reporting
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_ECONOMY_API_KEY,
    LLM_ECONOMY_BASE_URL,
    LLM_ECONOMY_FALLBACK_MODEL,
    LLM_ECONOMY_MODEL,
    LLM_FAST_MODEL,  # re-exported for fast-lane callers
    LLM_MODEL,
)

logger = logging.getLogger(__name__)

FAILOVER_STATUSES = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _lanes(lane: str = "generation") -> list[dict]:
    """Ordered provider lanes for the named lane; skip unconfigured ones.

    generation/fast share the classic chain - callers pass their model
    override (LLM_MODEL / LLM_FAST_MODEL) per call. Economy has its own
    chain with pinned models and never touches the fast lane's quota pool.
    """
    if lane == "economy":
        lanes = [
            {
                "name": "economy",
                "base_url": LLM_ECONOMY_BASE_URL,
                "api_key": LLM_ECONOMY_API_KEY,
                "model": LLM_ECONOMY_MODEL,
            },
            {
                # Deep free pool, distinct from gpt-oss-20b: cache warming
                # bursts must not starve the fast lane that routing needs.
                # qwen3 models emit <think> reasoning by default; the extra
                # payload switches it off (verified against Groq, Aug 2026).
                "name": "economy-fallback",
                "base_url": LLM_BASE_URL,
                "api_key": LLM_API_KEY,
                "model": LLM_ECONOMY_FALLBACK_MODEL,
                "payload_extra": {"reasoning_effort": "none"},
            },
        ]
        return [entry for entry in lanes if entry["api_key"]]

    lanes = [{"name": "primary", "base_url": LLM_BASE_URL, "api_key": LLM_API_KEY}]
    if (
        ENABLE_LLM_FALLBACK
        and FALLBACK_LLM_API_KEY
        and FALLBACK_LLM_BASE_URL != LLM_BASE_URL
    ):
        lanes.append({
            "name": "fallback",
            "base_url": FALLBACK_LLM_BASE_URL,
            "api_key": FALLBACK_LLM_API_KEY,
            "model": FALLBACK_LLM_MODEL,
        })
    return [entry for entry in lanes if entry["api_key"]]


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


def _resolve_model(lane: dict, model: str | None) -> str:
    return lane.get("model") or model or LLM_MODEL


def _is_failover(exc: LLMError) -> bool:
    return exc.status is None or exc.status in FAILOVER_STATUSES


def chat(messages: list[dict], temperature: float = 0.2,
         max_tokens: int | None = None, response_format: dict | None = None,
         model: str | None = None, lane: str = "generation") -> tuple[str, dict | None]:
    """Blocking completion across the named lane's chain; returns (content, usage).

    `response_format` (e.g. {"type": "json_object"}) is optional structured
    output; callers must tolerate providers that reject it.
    """
    last_error: LLMError | None = None
    for entry in _lanes(lane):
        try:
            return _chat_once(entry, messages, temperature, max_tokens, response_format, model)
        except LLMError as exc:
            if not _is_failover(exc):
                raise
            last_error = exc
            logger.warning("%s lane failed (%s); trying next lane", entry["name"], exc)
    raise last_error or LLMError(f"No LLM lane configured for {lane!r}")


def _chat_once(lane: dict, messages: list[dict], temperature: float,
               max_tokens: int | None, response_format: dict | None,
               model: str | None) -> tuple[str, dict | None]:
    url = f"{lane['base_url']}/chat/completions"
    payload: dict = {
        "model": _resolve_model(lane, model),
        "messages": messages,
        "temperature": temperature,
        **(lane.get("payload_extra") or {}),
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if response_format is not None:
        payload["response_format"] = response_format
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=_headers(lane["api_key"]), json=payload)
    except httpx.HTTPError as exc:
        raise LLMError(f"{lane['name']} lane unreachable: {exc}") from exc
    if resp.status_code != 200:
        raise LLMError(
            f"LLM HTTP {resp.status_code}: {resp.text[:300]}", status=resp.status_code
        )
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected LLM response shape: {data}") from exc
    return content or "", data.get("usage")


def stream_chat(messages: list[dict], temperature: float = 0.2,
                model: str | None = None, lane: str = "generation") -> Iterator[str]:
    """Streaming completion across the named lane's chain; yields content deltas.

    Falls over to the next lane only when nothing has been yielded yet - a
    mid-stream failure is surfaced to the caller as-is.
    """
    last_error: LLMError | None = None
    for entry in _lanes(lane):
        pieces = 0
        try:
            for piece in _stream_lane(entry, messages, temperature, model):
                pieces += 1
                yield piece
            return
        except LLMError as exc:
            if pieces or not _is_failover(exc):
                raise
            last_error = exc
            logger.warning("%s lane failed before first token (%s); trying next lane",
                           entry["name"], exc)
    raise last_error or LLMError(f"No LLM lane configured for {lane!r}")


def _stream_lane(lane: dict, messages: list[dict], temperature: float,
                 model: str | None) -> Iterator[str]:
    url = f"{lane['base_url']}/chat/completions"
    base_payload: dict = {
        "model": _resolve_model(lane, model),
        "messages": messages,
        "temperature": temperature,
        "stream": True,
        **(lane.get("payload_extra") or {}),
    }

    def _open(payload):
        try:
            return httpx.Client(timeout=120).stream(
                "POST", url, headers=_headers(lane["api_key"]), json=payload
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"{lane['name']} lane unreachable: {exc}") from exc

    def _gen(resp, usage: dict) -> Iterator[str]:
        if resp.status_code != 200:
            body = resp.read().decode(errors="replace")
            raise LLMError(f"LLM HTTP {resp.status_code}: {body[:300]}",
                           status=resp.status_code)
        for line in resp.iter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage.update(chunk["usage"])
            choices = chunk.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            content = delta.get("content")
            if content:
                yield content

    # Attempt with usage tracking; on HTTP 400 retry without it.
    usage: dict = {}
    try:
        payload = {**base_payload, "stream_options": {"include_usage": True}}
        with _open(payload) as resp:
            for piece in _gen(resp, usage):
                yield piece
            if usage:
                logger.info("%s stream usage: %s", lane["name"], usage)
            return
    except LLMError as exc:
        if exc.status != 400:
            raise
        logger.info("stream_options unsupported; retrying without")

    with _open(base_payload) as resp:
        for piece in _gen(resp, usage):
            yield piece
        if usage:
            logger.info("%s stream usage: %s", lane["name"], usage)
