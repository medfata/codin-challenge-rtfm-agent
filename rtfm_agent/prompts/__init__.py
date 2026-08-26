"""Versioned prompt templates + message builders.

Template texts live as .txt files next to this module (system, rewrite,
summary, route, chitchat, memory) so they can be reviewed and diffed in
PRs like any other artifact; this module loads them once at import and
exposes the builder functions that assemble full chat payloads.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """Load a prompt template by file name (without extension)."""
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# System prompts (loaded from templates)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = load_prompt("system")
REWRITE_SYSTEM = load_prompt("rewrite")
SUMMARY_SYSTEM = load_prompt("summary")
ROUTE_SYSTEM = load_prompt("route")
CHITCHAT_SYSTEM = load_prompt("chitchat")
MEMORY_SYSTEM = load_prompt("memory")

# ---------------------------------------------------------------------------
# Fixed reply / fragment strings
# ---------------------------------------------------------------------------

REFUSAL = (
    "I don't have enough information in the documentation to answer that question."
)

MEMORY_NONE_REPLY = (
    "I don't have any lasting memories about you yet. Tell me about your "
    "project or preferences and I'll remember them in future conversations."
)

REWRITE_SOURCE_LINE = (
    "\n\nAdditionally, if the conversation or the message references a specific "
    "document, guide, chapter, or file by name, append a final line exactly in "
    "the form:\nSOURCE: <best-matching file name>\nIf nothing specific is "
    "referenced, omit the SOURCE line."
)

CONTEXT_HEADER = "[source: {source_file} | {section_heading} | chunk {chunk_pos}]"

_SUMMARY_NOTE_PREFIX = "Earlier in this conversation: "

# ---------------------------------------------------------------------------
# Context rendering
# ---------------------------------------------------------------------------


def build_context(chunks) -> str:
    """Render retrieved chunks into a single numbered context string."""
    parts = []
    for i, c in enumerate(chunks, 1):
        header = CONTEXT_HEADER.format(
            source_file=c["source_file"],
            section_heading=c["section_heading"],
            chunk_pos=c["chunk_pos"],
        )
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def build_user_context_block(memories) -> str | None:
    """Render recalled long-term memories as a USER CONTEXT section."""
    if not memories:
        return None
    lines = []
    for m in memories:
        line = f"- {m['text']}"
        if m.get("topics"):
            line += f" [topics: {', '.join(m['topics'])}]"
        lines.append(line)
    return (
        "USER CONTEXT (durable facts about this user, remembered from "
        "previous conversations):\n" + "\n".join(lines)
    )


def compose_user_message(question: str, chunks, memories=None) -> str:
    """Assemble the final user message from all available context sources."""
    sections = []
    user_context = build_user_context_block(memories)
    if user_context:
        sections.append(user_context)
    if chunks:
        sections.append(f"CONTEXT:\n{build_context(chunks)}")
    sections.append(f"QUESTION: {question}")
    tail = (
        "\n\nAnswer using only the context above. Cite source paths for each claim."
        if chunks
        else "\n\nRespond naturally to the message above given what you know. If it shares durable information about the user or their project, acknowledge that you'll remember it."
    )
    return "\n\n".join(sections) + tail


# ---------------------------------------------------------------------------
# Single-turn builders (system + user payloads for one-shot LLM calls)
# ---------------------------------------------------------------------------


def build_rewrite_prompt(history_msgs, question: str) -> str:
    # History EXCLUDING the latest question; it is appended explicitly below
    # so the transcript ends with the actual user turn being rewritten.
    prior = history_msgs[:-1] if history_msgs else []
    convo = "\n".join(
        f"{m['role']}: {m['content'][:400]}" for m in prior
    )
    return (
        f"CONVERSATION SO FAR:\n{convo}\n\n"
        f"LATEST USER MESSAGE: {question}\n\n"
        "Task: rewrite LATEST USER MESSAGE as a standalone search query whose "
        "meaning is clear without the conversation. Reply with the query only."
        + REWRITE_SOURCE_LINE
    )


def build_summary_prompt(existing_summary: str | None, older_msgs) -> str:
    convo = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in older_msgs)
    prior = (
        f"EXISTING SUMMARY (fold this in):\n{existing_summary}\n\n" if existing_summary else ""
    )
    return (
        f"{prior}"
        f"OLDER MESSAGES TO SUMMARIZE:\n{convo}\n\n"
        "Produce the updated rolling summary now."
    )


def build_route_prompt(history_msgs, question: str) -> str:
    # History EXCLUDING the latest question; it is appended explicitly below.
    prior = history_msgs[:-1] if history_msgs else []
    convo = "\n".join(
        f"{m['role']}: {m['content'][:400]}" for m in prior
    )
    return (
        f"CONVERSATION SO FAR:\n{convo}\n\n"
        f"LATEST USER MESSAGE: {question}\n\n"
        "Classify LATEST USER MESSAGE and produce the intent JSON object now."
    )


def build_memory_prompt(memories, question: str) -> str:
    lines = [f"- {m['text']}" for m in memories]
    return (
        "REMEMBERED FACTS:\n" + "\n".join(lines)
        + f"\n\nUSER QUESTION: {question}"
    )


# ---------------------------------------------------------------------------
# Full chat-payload builders (used by the RAG pipeline and SSE stream)
# ---------------------------------------------------------------------------


def _with_summary(messages: list[dict], summary_note: str | None) -> list[dict]:
    if summary_note:
        messages.append({
            "role": "system",
            "content": f"{_SUMMARY_NOTE_PREFIX}{summary_note}",
        })
    return messages


def build_doc_messages(question: str, chunks, hist_turns: list[dict],
                       memories=None, summary_note: str | None = None) -> list[dict]:
    """Grounded-answer payload: system prompt + summary + history + context."""
    messages = _with_summary([{"role": "system", "content": SYSTEM_PROMPT}], summary_note)
    messages.extend(hist_turns)
    messages.append({"role": "user", "content": compose_user_message(question, chunks, memories)})
    return messages


def build_chitchat_messages(question: str, hist_turns: list[dict],
                            summary_note: str | None = None) -> list[dict]:
    """Casual-conversation payload on the chitchat persona."""
    messages = _with_summary([{"role": "system", "content": CHITCHAT_SYSTEM}], summary_note)
    messages.extend(hist_turns)
    messages.append({"role": "user", "content": question})
    return messages


def build_memory_messages(memories, question: str, hist_turns: list[dict],
                          summary_note: str | None = None) -> list[dict]:
    """Memory-synthesis payload over recalled durable facts."""
    messages = _with_summary([{"role": "system", "content": MEMORY_SYSTEM}], summary_note)
    messages.extend(hist_turns)
    messages.append({"role": "user", "content": build_memory_prompt(memories, question)})
    return messages
