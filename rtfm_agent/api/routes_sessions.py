"""Session-history endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from rtfm_agent.api import state
from rtfm_agent.common import sessions as sessions_mod
from rtfm_agent.common.tenancy import TenantContext, require_tenant

router = APIRouter()


@router.get("/sessions/{session_id}/history")
def get_session_history(session_id: str, limit: int = 50,
                        t: TenantContext = Depends(require_tenant)):
    r = state.get_redis()
    msgs = sessions_mod.history(r, t, session_id, last_n=limit)
    return {
        "session_id": session_id,
        "ttl_seconds": sessions_mod.ttl(r, t, session_id),
        "message_count": len(msgs),
        "summary": sessions_mod.get_summary(r, t, session_id),
        "messages": msgs,
    }


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, t: TenantContext = Depends(require_tenant)):
    removed = sessions_mod.clear(state.get_redis(), t, session_id)
    if not removed:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "deleted": True}
