"""Corpus directory resolution shared by the API, actions, MCP server, and
background workers (single source of truth for docs-dir precedence).

Precedence: docs/<tenant>/ > validated per-request override > default DOCS_DIR.
"""

from pathlib import Path

from fastapi import HTTPException

from rtfm_agent.common.tenancy import TenantContext
from rtfm_agent.config import settings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Repository root (contains docs/, static/, scripts/)."""
    return _PROJECT_ROOT


def resolve_docs_dir(t: TenantContext, override: str | None = None) -> Path:
    """Per-team corpus precedence: docs/<tenant>/ > validated override > DOCS_DIR."""
    team_dir = _PROJECT_ROOT / "docs" / t.id
    if team_dir.is_dir():
        return team_dir
    if override:
        resolved = Path(override).resolve()
        if not resolved.is_dir():
            raise HTTPException(
                status_code=422,
                detail=f"docs_dir '{override}' is not an existing directory",
            )
        try:
            resolved.relative_to(_PROJECT_ROOT)
        except ValueError:
            raise HTTPException(
                status_code=422, detail="docs_dir must stay inside the project root"
            )
        return resolved
    return Path(settings.docs.dir)


def docs_dir_for(t: TenantContext) -> str:
    """This tenant's default corpus directory, for version/drift checks."""
    return str(resolve_docs_dir(t))
