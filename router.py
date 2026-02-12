"""
Smart Query Router Module.
Determines whether a user query should be handled by document RAG,
web search, both, or as a general conversation.
"""

import re
import logging

logger = logging.getLogger(__name__)


# ─── Keyword Patterns ────────────────────────────────────────────────────────
WEB_KEYWORDS = {
    "latest", "recent", "news", "today", "current", "trending",
    "update", "2024", "2025", "2026", "live", "now", "breaking",
    "weather", "stock", "price", "score", "result",
}

DOC_KEYWORDS = {
    "document", "file", "uploaded", "pdf", "page", "paragraph",
    "section", "chapter", "table", "figure", "according to",
    "in the file", "in the document", "my file", "my document",
    "the report", "the paper", "mentioned", "states", "says",
}

GENERAL_KEYWORDS = {
    "hello", "hi", "hey", "thanks", "thank you", "bye", "help",
    "who are you", "what can you do", "how are you",
}


class QueryRouter:
    """Routes user queries to the appropriate data source."""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional LLMClient for LLM-based classification.
        """
        self.llm_client = llm_client

    def route(self, query: str, has_documents: bool) -> str:
        """
        Determine the routing strategy for a query.

        Args:
            query: The user's query.
            has_documents: Whether documents are loaded in the vector store.

        Returns:
            One of: 'DOCS_ONLY', 'WEB_ONLY', 'BOTH', 'GENERAL'.
        """
        query_lower = query.lower().strip()

        # 1. Quick check for greetings / general chat
        if self._is_general(query_lower):
            logger.info(f"Router: GENERAL (keyword match)")
            return "GENERAL"

        # 2. Keyword-based heuristics
        has_web_signal = self._has_keywords(query_lower, WEB_KEYWORDS)
        has_doc_signal = self._has_keywords(query_lower, DOC_KEYWORDS)

        if has_doc_signal and not has_web_signal:
            route = "DOCS_ONLY" if has_documents else "WEB_ONLY"
        elif has_web_signal and not has_doc_signal:
            route = "WEB_ONLY"
        elif has_web_signal and has_doc_signal:
            route = "BOTH" if has_documents else "WEB_ONLY"
        else:
            # 3. Ambiguous — use LLM classification if available
            if self.llm_client:
                route = self.llm_client.classify_query(query, has_documents)
            else:
                # Default: search docs if available, else general
                route = "DOCS_ONLY" if has_documents else "GENERAL"

        logger.info(f"Router: {route} for query '{query[:50]}...'")
        return route

    # ─── Private Methods ─────────────────────────────────────────────────

    def _is_general(self, query: str) -> bool:
        """Check if the query is a general greeting or meta-question."""
        for keyword in GENERAL_KEYWORDS:
            if query.startswith(keyword) or query == keyword:
                return True
        return len(query.split()) <= 2 and not any(
            c in query for c in "?."
        )

    def _has_keywords(self, query: str, keywords: set) -> bool:
        """Check if the query contains any of the given keywords."""
        query_words = set(re.findall(r"\w+", query))
        return bool(query_words & keywords)
