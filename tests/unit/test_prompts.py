"""Prompt builder tests: templates load and message payloads assemble."""

from rtfm_agent.prompts import (
    CHITCHAT_SYSTEM,
    MEMORY_NONE_REPLY,
    REFUSAL,
    ROUTE_SYSTEM,
    REWRITE_SYSTEM,
    SUMMARY_SYSTEM,
    SYSTEM_PROMPT,
    build_chitchat_messages,
    build_context,
    build_doc_messages,
    build_memory_messages,
    build_rewrite_prompt,
    build_route_prompt,
    build_summary_prompt,
    compose_user_message,
)


def test_all_templates_loaded():
    for template in (SYSTEM_PROMPT, REWRITE_SYSTEM, SUMMARY_SYSTEM,
                     ROUTE_SYSTEM, CHITCHAT_SYSTEM):
        assert isinstance(template, str) and len(template) > 50


def test_fixed_replies_present():
    assert "documentation" in REFUSAL
    assert "memories" in MEMORY_NONE_REPLY


def test_build_context_includes_headers(chunks=None):
    chunks = [{
        "source_file": "book/ch1.asc",
        "section_heading": "Basics",
        "chunk_pos": 3,
        "text": "Body text here.",
    }]
    ctx = build_context(chunks)
    assert "book/ch1.asc" in ctx
    assert "Basics" in ctx
    assert "Body text here." in ctx


def test_compose_user_message_orders_sections():
    chunks = [{"source_file": "a.asc", "section_heading": "H", "chunk_pos": 1, "text": "T"}]
    memories = [{"text": "uses Go", "topics": ["go"]}]
    msg = compose_user_message("what is git?", chunks, memories)
    assert msg.index("USER CONTEXT") < msg.index("CONTEXT:") < msg.index("QUESTION:")


def test_doc_message_roles():
    msgs = build_doc_messages(
        "q?", [], [{"role": "user", "content": "hi"}], None, None
    )
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "q?" in msgs[-1]["content"]


def test_summary_note_injected_as_system():
    msgs = build_chitchat_messages("hello", [], summary_note="earlier stuff")
    assert any(m["role"] == "system" and "earlier stuff" in m["content"]
               for m in msgs)


def test_memory_messages_reference_facts():
    msgs = build_memory_messages([{"text": "likes Rust"}], "fav lang?", [], None)
    joined = "\n".join(m["content"] for m in msgs)
    assert "likes Rust" in joined


def test_rewrite_prompt_excludes_latest_from_history():
    hist = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "LATEST"},
    ]
    prompt = build_rewrite_prompt(hist, "LATEST")
    assert "LATEST USER MESSAGE: LATEST" in prompt
    assert "first" in prompt


def test_route_prompt_shape():
    prompt = build_route_prompt([], "show my stats")
    assert "Classify LATEST USER MESSAGE" in prompt


def test_summary_prompt_folds_existing():
    prompt = build_summary_prompt("old summary", [{"role": "user", "content": "msg"}])
    assert "EXISTING SUMMARY" in prompt
