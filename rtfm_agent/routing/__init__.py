"""Agent intelligence layer: intent classification, conversational actions,
long-term memory client, and semantic-cache warming."""

from rtfm_agent.routing import intent
from rtfm_agent.routing.actions import dispatch
from rtfm_agent.routing.intent import RouteResult, classify

__all__ = ["RouteResult", "actions", "classify", "dispatch", "intent"]
