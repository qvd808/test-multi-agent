"""Search tool — Tavily API wrapper for the researcher agent.

Provides a clean interface for searching and retrieving structured results
from the Tavily web search API.
"""

from __future__ import annotations

import os

from tavily import TavilyClient


def search(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
) -> list[dict]:
    """Search Tavily and return structured results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        search_depth: "basic" or "advanced" (default).

    Returns:
        List of dicts with keys: title, content, url, score.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
    )

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "url": r.get("url", ""),
            "score": r.get("score", 0.0),
        })

    return results


def search_context(query: str, max_results: int = 5) -> str:
    """Search and return results as a formatted markdown string.

    Convenience wrapper for agents that want plain text context.
    """
    results = search(query, max_results=max_results)
    chunks = []
    for r in results:
        chunks.append(f"### {r['title']}\n{r['content']}\n*Source: {r['url']}*")
    return "\n\n---\n\n".join(chunks) if chunks else "(No results found)"
