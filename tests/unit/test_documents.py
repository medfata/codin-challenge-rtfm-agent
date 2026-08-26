"""Document loader unit tests: heading extraction and file discovery."""

from rtfm_agent.ingestion.documents import extract_heading, load_asc_files


def test_extract_asciidoc_heading():
    assert extract_heading("== Introduction\n\ntext") == "Introduction"
    assert extract_heading("=== Sub-section\n\ntext") == "Sub-section"


def test_extract_markdown_heading():
    assert extract_heading("# Title\n\ntext") == "Title"
    assert extract_heading("## not h1") is None


def test_extract_heading_missing():
    assert extract_heading("just text, no headings") is None


def test_load_asc_files_discovers_and_shapes(tmp_path):
    (tmp_path / "a.asc").write_text("== Alpha\n\ncontent", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.asc").write_text("== Beta\n\ncontent", encoding="utf-8")

    top = load_asc_files(str(tmp_path))
    assert [d["source_file"] for d in top] == ["a.asc"]
    assert top[0]["heading"] == "Alpha"

    recursive = load_asc_files(str(tmp_path), recursive=True)
    assert sorted(d["source_file"] for d in recursive) == ["a.asc", "sub/b.asc"]


def test_load_respects_extension_filter(tmp_path):
    (tmp_path / "doc.md").write_text("# Markdown doc", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("not a doc", encoding="utf-8")
    docs = load_asc_files(str(tmp_path))
    assert [d["source_file"] for d in docs] == ["doc.md"]
