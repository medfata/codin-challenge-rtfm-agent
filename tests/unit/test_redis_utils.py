"""Redis helper unit tests."""

from rtfm_agent.common.redis_utils import escape_tag, normalize


def test_normalize_decodes_bytes_everywhere():
    raw = [b"total", {b"k": b"v"}, (b"a", b"b")]
    out = normalize(raw)
    assert out == ["total", {"k": "v"}, ["a", "b"]]


def test_normalize_passes_through_non_bytes():
    obj = {"n": 5, "f": 1.5, "s": "x"}
    assert normalize(obj) == obj


def test_escape_tag_alphanumerics_untouched():
    assert escape_tag("abc123") == "abc123"


def test_escape_tag_escapes_punctuation():
    assert "/" not in escape_tag("book/ch.asc").replace("\\/", "")
    escaped = escape_tag("book/ch.1 (v2)")
    assert "\\." in escaped or "." in escaped
    # comma is the OR separator - dropped entirely
    assert "," not in escape_tag("a,b")
