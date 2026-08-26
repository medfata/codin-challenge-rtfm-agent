"""Chunker unit tests: paragraph-aware splitting with overlap."""

from rtfm_agent.ingestion.chunker import do_chunk_text, load_and_chunk_docs


def test_short_content_single_chunk():
    chunks = list(do_chunk_text("One short paragraph about Git."))
    assert len(chunks) == 1
    pos, text = chunks[0]
    assert pos == 1
    assert "Git" in text


def test_empty_content_yields_nothing():
    assert list(do_chunk_text("")) == []
    assert list(do_chunk_text("\n\n\n")) == []


def test_paragraphs_split_into_chunks():
    para = "Paragraph %d with some content words here."
    content = "\n\n".join(para % i for i in range(200))
    chunks = list(do_chunk_text(content, target_chars=400))
    assert len(chunks) > 1
    positions = [p for p, _ in chunks]
    assert positions == sorted(positions)
    assert positions[0] == 1


def test_chunk_positions_are_sequential():
    content = "\n\n".join(f"para {i} " + "word " * 80 for i in range(40))
    chunks = list(do_chunk_text(content, target_chars=300))
    assert [p for p, _ in chunks] == list(range(1, len(chunks) + 1))


def test_load_and_chunk_docs_shape(sample_docs_dir):
    from rtfm_agent.ingestion.documents import load_asc_files

    docs = load_asc_files(str(sample_docs_dir))
    chunks = load_and_chunk_docs(docs)
    assert chunks, "expected chunks"
    for c in chunks:
        assert {"source_file", "heading", "chunk_pos", "chunk_text"} <= set(c)


def test_max_chunks_per_doc_limit(sample_docs_dir):
    from rtfm_agent.ingestion.documents import load_asc_files

    docs = load_asc_files(str(sample_docs_dir))
    limited = load_and_chunk_docs(docs, max_chunks_per_doc=1)
    assert all(c["chunk_pos"] <= 1 for c in limited)
