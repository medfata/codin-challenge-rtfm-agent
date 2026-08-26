"""Query-time retrieval: KNN search, semantic cache, scope detection,
query rewriting, and the RAG pipeline orchestration."""

from rtfm_agent.retrieval import cache, citations, scope, search
from rtfm_agent.retrieval.rag import answer_question
from rtfm_agent.retrieval.rewrite import rewrite_query, summarize_turns

__all__ = [
    "answer_question",
    "cache",
    "citations",
    "rewrite_query",
    "scope",
    "search",
    "summarize_turns",
]
