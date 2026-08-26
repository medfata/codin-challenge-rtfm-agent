"""Citation dedupe unit tests (shared by RAG pipeline and cache warming)."""

from rtfm_agent.retrieval.citations import dedupe_citations


def _chunk(src, heading, pos, score):
    return {"source_file": src, "section_heading": heading,
            "chunk_pos": pos, "score": score, "text": "..."}


def test_best_score_wins_per_source():
    chunks = [
        _chunk("a.asc", "H1", 1, 0.30),
        _chunk("a.asc", "H1", 2, 0.20),
        _chunk("b.asc", "H2", 1, 0.25),
    ]
    out = dedupe_citations(chunks)
    # sorted by ascending cosine distance: a@0.20, then b@0.25
    assert [c["source_file"] for c in out] == ["a.asc", "b.asc"]
    assert out[0]["score"] == 0.2
    assert out[0]["chunk_pos"] == 2


def test_scores_rounded_and_sorted():
    out = dedupe_citations([_chunk("x.asc", "H", 1, 0.123456)])
    assert out[0]["score"] == 0.1235


def test_empty_input():
    assert dedupe_citations([]) == []
