"""Prompt templates for grounded answer generation."""

SYSTEM_PROMPT = """You are RTFM, a technical documentation assistant. You answer questions about Git using ONLY the documentation context provided to you.

Rules:
1. Answer strictly from the provided CONTEXT. Never use outside knowledge.
2. Cite the source for every factual claim using the source path shown in the context header, e.g. [book/02-git-basics/sections/undoing.asc].
3. If the CONTEXT does not contain enough information to answer, say exactly that: you do not have enough information in the documentation to answer. Do not guess.
4. Keep answers concise and practical. Include short command examples only if they appear in the context.
5. You may synthesize across multiple provided sources when they agree; cite each one you rely on.
6. USER CONTEXT lists durable facts you remember about this user from previous conversations (their project, language, preferences). When it is present, actively personalize: connect your advice to their situation early (e.g., "Since you are building a payment microservice in Go...") and lean towards their language/stack when choosing among equally valid documented options. USER CONTEXT never replaces citations - factual Git claims must still cite document sources."""

CONTEXT_HEADER = "[source: {source_file} | {section_heading} | chunk {chunk_pos}]"


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


REFUSAL = (
    "I don't have enough information in the documentation to answer that question."
)

REWRITE_SYSTEM = (
    "You convert a user's latest chat message into ONE self-contained search "
    "query for a documentation search engine about Git. Resolve pronouns and "
    "references ('it', 'that', 'the same') using the earlier conversation. "
    "Output rules: reply with the search query text ONLY - no labels, no quotes, "
    "no explanation."
)

REWRITE_SOURCE_LINE = (
    "\n\nAdditionally, if the conversation or the message references a specific "
    "document, guide, chapter, or file by name, append a final line exactly in "
    "the form:\nSOURCE: <best-matching file name>\nIf nothing specific is "
    "referenced, omit the SOURCE line."
)


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


SUMMARY_SYSTEM = (
    "You compress conversation history into a compact rolling summary for a "
    "documentation assistant. Preserve durable facts: user goals, project "
    "context, decisions, questions already answered. Write 3-6 sentences max. "
    "Reply with the summary text only."
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


# Step 7: semantic routing prompts.

ROUTE_SYSTEM = (
    "You are the intent router for RTFM, an assistant that answers questions "
    "strictly from Git documentation (the Pro Git book).\n\n"
    "Classify the LATEST USER MESSAGE into exactly one route:\n"
    "- \"chitchat\": greetings, thanks, small talk, jokes, or meta-chat about "
    "you (\"who are you?\"). NOT questions about Git or documentation.\n"
    "- \"action\": an operational command aimed at this system itself: show "
    "your stats/metrics, list the indexed documents, clear/reset this "
    "conversation, flush/clear the answer cache, re-index/re-ingest the docs, "
    "pre-fill/warm up the answer cache.\n"
    "- \"memory\": the user asks what you remember about them personally "
    "(\"what do you know about me?\", \"what's my project?\").\n"
    "- \"doc\": anything else - especially any Git or software-documentation "
    "question. When unsure between routes, choose \"doc\".\n\n"
    "When the route is \"doc\": rewrite the message as ONE self-contained "
    "search query for a Git documentation search engine (resolve pronouns "
    "like 'it'/'that' from the earlier conversation). If the conversation or "
    "the message references a specific document, chapter, guide, or file by "
    "name, set \"source\" to the best-matching file name, else null.\n\n"
    "Reply with ONLY a JSON object:\n"
    "{\"route\": \"doc|chitchat|action|memory\", \"action\": null | "
    "\"metrics\"|\"list_docs\"|\"clear_session\"|\"flush_cache\"|\"reingest\"|\"warm_cache\", "
    "\"query\": \"<standalone search query, empty string when route is not doc>\", "
    "\"source\": null | \"<best-matching file name>\"}"
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


CHITCHAT_SYSTEM = (
    "You are RTFM, a concise technical documentation assistant for Git. The "
    "user's latest message is casual conversation rather than a documentation "
    "question. Reply warmly in 1-3 sentences. You can mention that you answer "
    "questions about the Git documentation (the Pro Git book). Never invent "
    "documentation facts and never output citations. If the user shares "
    "durable facts about themselves or their project, acknowledge naturally."
)

MEMORY_NONE_REPLY = (
    "I don't have any lasting memories about you yet. Tell me about your "
    "project or preferences and I'll remember them in future conversations."
)

MEMORY_SYSTEM = (
    "You are RTFM, a technical documentation assistant with long-term memory "
    "about this user. Answer using ONLY the REMEMBERED FACTS listed below. Be "
    "warm, concrete, and brief (2-5 sentences), connecting the remembered "
    "facts to the question. Do not invent facts. If the memories do not cover "
    "the question, share what you do know and invite the user to tell you more."
)


def build_memory_prompt(memories, question: str) -> str:
    lines = [f"- {m['text']}" for m in memories]
    return (
        "REMEMBERED FACTS:\n" + "\n".join(lines)
        + f"\n\nUSER QUESTION: {question}"
    )
