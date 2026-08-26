"""Intent parser unit tests: defensive JSON handling without any LLM call."""

from rtfm_agent.routing.intent import RouteResult, _parse


def test_valid_doc_route_with_query():
    raw = '{"route":"doc","query":"what is git rebase","source":null}'
    r = _parse(raw, original="orig question")
    assert r.route == "doc"
    assert r.query == "what is git rebase"


def test_short_query_falls_back_to_original():
    raw = '{"route":"doc","query":"hi"}'
    r = _parse(raw, original="the original long enough question")
    assert r.query == "the original long enough question"


def test_chitchat_route_ignores_query():
    raw = '{"route":"chitchat","query":"hello there friend"}'
    r = _parse(raw, original="hey")
    assert r.route == "chitchat" and r.query == "hey"


def test_action_route_requires_known_verb():
    raw = '{"route":"action","action":"metrics"}'
    assert _parse(raw, "show stats").action == "metrics"

    bad = '{"route":"action","action":"delete_everything"}'
    r = _parse(bad, "delete everything")
    assert r.route == "doc"  # degrades


def test_destructive_action_guard_rejects_without_keyword():
    # LLM says flush_cache but the raw text never mentions the cache.
    raw = '{"route":"action","action":"flush_cache"}'
    r = _parse(raw, original="please do the thing")
    assert r.route == "doc"


def test_destructive_action_passes_with_keyword():
    raw = '{"route":"action","action":"flush_cache"}'
    r = _parse(raw, original="please flush the cache")
    assert r.route == "action" and r.action == "flush_cache"


def test_malformed_json_degrades_to_doc():
    for raw in ("not json", '{"route":', "", "[]"):
        r = _parse(raw, original="fallback question")
        assert r == RouteResult(query="fallback question")


def test_unknown_route_degrades():
    r = _parse('{"route":"teleport"}', original="q")
    assert r.route == "doc"


def test_source_hint_extracted():
    raw = '{"route":"doc","query":"undoing commits guide","source":"undoing.asc"}'
    r = _parse(raw, original="x")
    assert r.source_hint == "undoing.asc"
