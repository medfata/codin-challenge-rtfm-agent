"""Heuristic RAG quality metrics - no external eval framework required.

Each metric scores one (question, answer, citations) triple against its
dataset expectations. Scores are deliberately simple and explainable:
they catch regressions in grounding behaviour, not subtle semantics.
"""

from __future__ import annotations


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer (1.0 when none)."""
    if not keywords:
        return 1.0
    lowered = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in lowered)
    return hits / len(keywords)


def has_citations(citations: list) -> bool:
    return bool(citations)


def is_refusal(answer: str) -> bool:
    """Match the assistant's documented refusal phrasing."""
    lowered = answer.lower()
    return (
        "don't have enough information" in lowered
        or "do not have enough information" in lowered
    )


def score_example(example: dict, result: dict) -> dict:
    """Evaluate one dataset row against one pipeline response.

    `example`: dataset row (question, expected_keywords, must_cite,
               expect_refusal).
    `result`:  API response fields (answer, citations, route, ...).
    Returns {passed, checks: [{name, passed, detail}]}.
    """
    answer = result.get("answer", "")
    citations = result.get("citations") or []
    checks = []

    kw = example.get("expected_keywords") or []
    coverage = keyword_coverage(answer, kw)
    checks.append({
        "name": "keyword_coverage",
        "passed": coverage >= 0.5 if not example.get("expect_refusal") else True,
        "detail": f"{coverage:.0%} of {len(kw)} keywords",
    })

    if example.get("expect_refusal"):
        refused = is_refusal(answer)
        checks.append({
            "name": "refuses_out_of_scope",
            "passed": refused,
            "detail": "refused" if refused else f"answered anyway: {answer[:80]!r}",
        })
    else:
        checks.append({
            "name": "answers_question",
            "passed": not is_refusal(answer),
            "detail": "answered" if not is_refusal(answer) else "unexpected refusal",
        })

    if example.get("must_cite"):
        cited = has_citations(citations)
        checks.append({
            "name": "cites_sources",
            "passed": cited,
            "detail": f"{len(citations)} citation(s)",
        })

    return {
        "id": example.get("id"),
        "question": example.get("question"),
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
