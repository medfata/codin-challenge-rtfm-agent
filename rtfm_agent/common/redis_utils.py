"""Shared Redis response helpers (single source of truth for all modules)."""


def normalize(obj):
    """Recursively decode redis-py's bytes-keyed FT.SEARCH/FT.AGGREGATE reply."""
    if isinstance(obj, bytes):
        return obj.decode(errors="replace")
    if isinstance(obj, dict):
        return {normalize(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(x) for x in obj]
    return obj


def escape_tag(value: str) -> str:
    """Escape punctuation for use inside a RediSearch TAG query set."""
    out = []
    for ch in value:
        if ch.isalnum():
            out.append(ch)
        elif ch == ",":
            continue  # comma is the OR separator inside {...}
        else:
            out.append("\\" + ch)
    return "".join(out)
