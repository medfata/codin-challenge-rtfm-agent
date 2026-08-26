"""Citation shaping shared by the RAG pipeline and cache warming.

Lives in its own neutral module so retrieval and routing can both import it
without creating an import cycle.
"""


def dedupe_citations(chunks) -> list[dict]:
    """Best citation per (source_file, section_heading), sorted by distance."""
    deduped: dict[tuple[str, str], dict] = {}
    for c in chunks:
        key = (c["source_file"], c["section_heading"])
        if key not in deduped or c["score"] < deduped[key]["score"]:
            deduped[key] = {
                "source_file": c["source_file"],
                "section_heading": c["section_heading"],
                "chunk_pos": c["chunk_pos"],
                "score": round(c["score"], 4),
            }
    return sorted(deduped.values(), key=lambda x: x["score"])
