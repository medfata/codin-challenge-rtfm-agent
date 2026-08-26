"""Backward-compatible shim; implementation lives in rtfm_agent.documents."""

from rtfm_agent.ingestion.documents import extract_heading, load_asc_files

__all__ = ["extract_heading", "load_asc_files"]

if __name__ == "__main__":
    docs = load_asc_files("docs/progit2", recursive=True)
    print(f"Loaded {len(docs)} documentation files")
    for doc in docs[:5]:
        print(f'  - {doc["source_file"]}: heading={doc["heading"]!r}')
    if len(docs) > 5:
        print(f"  ... and {len(docs) - 5} more")
