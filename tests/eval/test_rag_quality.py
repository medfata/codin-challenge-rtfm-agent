"""Eval-metric unit tests: scoring logic on canned responses."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.metrics.rag_quality import (  # noqa: E402
    is_refusal,
    keyword_coverage,
    score_example,
)


def test_keyword_coverage_full_and_partial():
    assert keyword_coverage("git reset soft keeps changes", ["reset"]) == 1.0
    assert keyword_coverage("nothing relevant", ["reset", "stash"]) == 0.0
    assert keyword_coverage("anything", []) == 1.0


def test_is_refusal_matches_documented_phrasing():
    assert is_refusal(
        "I don't have enough information in the documentation to answer that question."
    )
    assert not is_refusal("Git is a version control system.")


def test_score_example_passes_grounded_answer():
    example = {
        "id": "t1",
        "question": "q",
        "expected_keywords": ["rebase"],
        "must_cite": True,
        "expect_refusal": False,
    }
    result = {
        "answer": "Rebase replays commits on another branch.",
        "citations": [{"source_file": "a.asc"}],
    }
    verdict = score_example(example, result)
    assert verdict["passed"]


def test_score_example_flags_missing_citations():
    example = {"id": "t2", "question": "q", "expected_keywords": [],
               "must_cite": True, "expect_refusal": False}
    result = {"answer": "An answer without sources.", "citations": []}
    verdict = score_example(example, result)
    assert not verdict["passed"]
    assert any(c["name"] == "cites_sources" and not c["passed"]
               for c in verdict["checks"])


def test_score_example_requires_refusal():
    example = {"id": "t3", "question": "off-topic", "expected_keywords": [],
               "must_cite": False, "expect_refusal": True}
    refused = score_example(example, {
        "answer": "I don't have enough information in the documentation.",
        "citations": [],
    })
    assert refused["passed"]

    hallucinated = score_example(example, {
        "answer": "Brazil won the 1998 World Cup final.",
        "citations": [],
    })
    assert not hallucinated["passed"]
