"""
Web Search Module.
Integrates Tavily API for real-time web search to augment RAG responses.
"""

import logging
from typing import Optional

from tavily import TavilyClient

from config import TAVILY_API_KEY, MAX_SEARCH_RESULTS

logger = logging.getLogger(__name__)


class WebSearcher:
    """Tavily-powered web search for retrieving real-time information."""

    def __init__(self):
        """Initialize the Tavily client."""
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is not set. Check your .env file.")
        self.client = TavilyClient(api_key=TAVILY_API_KEY)
        logger.info("WebSearcher initialized (Tavily API).")

    def search(
        self,
        query: str,
        max_results: int = MAX_SEARCH_RESULTS,
    ) -> list[dict]:
        """
        Search the web for a given query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of dicts with keys: 'title', 'url', 'content'.
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=False,
            )

            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", "No Title"),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                })

            logger.info(
                f"Web search for '{query[:50]}' returned {len(results)} results."
            )
            return results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def format_results(self, results: list[dict]) -> str:
        """
        Format search results into a readable string for the LLM.

        Args:
            results: List of search result dicts.

        Returns:
            Formatted string with numbered results.
        """
        if not results:
            return ""

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"**[{i}] {result['title']}**\n"
                f"URL: {result['url']}\n"
                f"{result['content']}\n"
            )

        return "\n---\n".join(formatted)
