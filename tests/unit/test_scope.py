"""Scope-detection pure-logic tests: hint regex + fuzzy file matching."""

from rtfm_agent.retrieval.scope import detect_doc_hint, match_files


INVENTORY = [
    "book/01-getting-started.asc",
    "book/02-git-basics/sections/undoing.asc",
    "book/03-branching.asc",
]


def test_detect_hint_from_question():
    q = "Based on the getting started guide, how do I configure git?"
    assert detect_doc_hint(q) == "getting started"


def test_no_hint_when_absent():
    assert detect_doc_hint("how do I rebase?") is None
    assert detect_doc_hint("in this one what does it say") is None


def test_match_files_ranks_by_overlap():
    files = match_files("getting started", INVENTORY)
    assert files[0] == "book/01-getting-started.asc"


def test_match_files_top_n():
    files = match_files("git book", INVENTORY, top_n=1)
    assert len(files) == 1


def test_match_files_empty_hint():
    assert match_files("", INVENTORY) == []
