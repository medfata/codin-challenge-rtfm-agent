"""Multi-tenant context: identity validation and per-tenant Redis naming.

Every request (except /health) must carry an X-Tenant-Id header. The id is
validated against a slug pattern and the TENANTS allowlist, then wrapped in a
TenantContext that derives all per-tenant key/index names:

    t:{org}:doc:{file}:{pos}       t:{org}:doc_idx
    t:{org}:cache:{uuid}           t:{org}:cache_idx
    t:{org}:session:{sid}:{...}    t:{org}:metrics:cache
    t:{org}:corpus                 t:{org}:docmeta:{file}

The Agent Memory Server is scoped via namespace={org} instead of key prefixes
(see routing/memory.py). The strict-slug validation prevents key-space
injection: a tenant id can never contain ':' or wildcard characters.
"""

import logging
import re

from fastapi import Header, HTTPException

from rtfm_agent.config import settings

logger = logging.getLogger(__name__)

_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9-_]{0,62}$")


class TenantContext:
    """Validated tenant identity plus its derived Redis naming scheme."""

    __slots__ = ("id",)

    def __init__(self, tenant_id: str):
        self.id = tenant_id

    @property
    def prefix(self) -> str:
        return f"t:{self.id}:"

    @property
    def doc_index(self) -> str:
        return f"{self.prefix}doc_idx"

    @property
    def cache_index(self) -> str:
        return f"{self.prefix}cache_idx"

    @property
    def metrics_key(self) -> str:
        return f"{self.prefix}metrics:cache"

    def __repr__(self) -> str:
        return f"TenantContext({self.id!r})"


def normalize_tenant(raw: str | None) -> TenantContext | None:
    """Validate a raw tenant id; None when missing/malformed/not allowed."""
    if not raw:
        return None
    tenant_id = raw.strip().lower()
    if not _TENANT_RE.match(tenant_id):
        return None
    if not settings.tenants.open and tenant_id not in settings.tenants.allowlist:
        return None
    return TenantContext(tenant_id)


def resolve_tenant(raw: str | None) -> TenantContext:
    """Validate a raw tenant id or raise the API's 422/403 errors.

    Shared by require_tenant (X-Tenant-Id header) and the SSE feed's
    ?tenant= query param (EventSource cannot set custom headers).
    """
    ctx = normalize_tenant(raw)
    if ctx is None:
        stripped = (raw or "").strip()
        if not stripped or not _TENANT_RE.match(stripped.lower()):
            raise HTTPException(
                status_code=422,
                detail="missing or invalid X-Tenant-Id header "
                       "(expected slug [a-z0-9][a-z0-9-_]{0,62})",
            )
        raise HTTPException(
            status_code=403,
            detail=f"tenant '{stripped.lower()}' is not in the allowed tenants list",
        )
    return ctx


def require_tenant(x_tenant_id: str = Header(default="")) -> TenantContext:
    """FastAPI dependency resolving the X-Tenant-Id header.

    422 when missing or malformed, 403 when not in the TENANTS allowlist.
    """
    return resolve_tenant(x_tenant_id)
