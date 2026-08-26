"""Config loading tests: sections exist and env aliases still resolve."""

from rtfm_agent.config import settings


def test_sections_present():
    for section in ("redis", "embed", "docs", "llm", "retrieval", "cache",
                    "sessions", "memory", "pricing", "routing", "versioning",
                    "mcp", "events", "warm", "tenants", "crawl"):
        assert hasattr(settings, section), section


def test_defaults_match_documented_env_names():
    assert settings.redis.url.startswith("redis://")
    assert settings.embed.dim == 384
    assert settings.retrieval.max_distance == 0.40
    assert settings.cache.threshold == 0.15
    assert settings.crawl.hard_page_cap == 500
    assert settings.events.heartbeat_s == 15


def test_llm_key_derivation_chain():
    from rtfm_agent.config import LlmSettings

    s = LlmSettings(google_api_key="g-key", api_key="", fallback_api_key="",
                    economy_api_key="")
    assert s.fallback_api_key == "g-key"
    assert s.economy_api_key == "g-key"

    s2 = LlmSettings(api_key="groq-key", google_api_key="g-key",
                     economy_api_key="econ-key", fallback_api_key="")
    assert s2.fallback_api_key == "g-key"
    assert s2.economy_api_key == "econ-key"


def test_tenants_allowlist_parsing():
    from rtfm_agent.config import TenantsSettings

    open_t = TenantsSettings(raw="*")
    assert open_t.open and "*" in open_t.allowlist

    closed = TenantsSettings(raw="acme,globex")
    assert not closed.open
    assert closed.allowlist == frozenset({"acme", "globex"})


def test_csv_list_parsing():
    from rtfm_agent.config import EventsSettings, McpSettings

    assert McpSettings(allowed_hosts="a.com,b.com").allowed_hosts == ["a.com", "b.com"]
    assert EventsSettings(cors_origins="*").cors_origins == ["*"]
