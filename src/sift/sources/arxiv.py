from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import feedparser
import httpx

from sift.sources.base import Article, Source

ARXIV_API = "https://export.arxiv.org/api/query"


class ArxivSource(Source):
    """arXiv search via the export API.

    Composes a list of category codes (e.g. cs.AI, cs.LG) with an optional
    free-text keyword query into a single search_query. Lets the user filter
    "papers in these subfields about these topics" — something the plain
    arxiv.org/rss/<cat> per-category feeds can't do.

    `query` is matched against title + abstract + comments via arXiv's
    `all:` field. Multiple words are AND-ed by arXiv's default scoring.
    """

    def __init__(
        self,
        *,
        id: str,
        categories: list[str] | None = None,
        query: str = "",
        max_results: int = 20,
        cadence_seconds: int = 21600,
    ) -> None:
        self.id = id
        self.categories = categories or []
        self.query = (query or "").strip()
        self.max_results = max_results
        self.cadence_seconds = cadence_seconds
        if not self.categories and not self.query:
            self.disabled = True
            self.disabled_reason = "no categories or query specified"

    async def poll(self) -> list[Article]:
        if self.disabled:
            return []
        params = {
            "search_query": self._build_search_query(),
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(ARXIV_API, params=params)
            resp.raise_for_status()
            xml = resp.text
        # feedparser is sync and CPU-ish — run in a thread to keep the loop free.
        feed = await asyncio.to_thread(feedparser.parse, xml)
        out: list[Article] = []
        for entry in feed.entries:
            url = getattr(entry, "link", None)
            title = (getattr(entry, "title", "") or "").replace("\n", " ").strip()
            if not url or not title:
                continue
            summary = (getattr(entry, "summary", "") or "").replace("\n", " ").strip()
            authors = getattr(entry, "authors", []) or []
            author_names = ", ".join(a.get("name", "") for a in authors[:3] if a.get("name"))
            out.append(
                Article(
                    source_id=self.id,
                    url=url,
                    title=title,
                    body=summary,
                    author=author_names or None,
                    posted_at=_parse_published(entry),
                )
            )
        return out

    def _build_search_query(self) -> str:
        """Build the arXiv search_query expression. Categories are OR-joined
        and AND-ed with the keyword clause. Returns 'all:*' as a no-op
        fallback (caller already disables empty configurations)."""
        parts: list[str] = []
        if self.categories:
            cat_clause = " OR ".join(f"cat:{c}" for c in self.categories)
            if len(self.categories) > 1:
                cat_clause = f"({cat_clause})"
            parts.append(cat_clause)
        if self.query:
            parts.append(f"all:({self.query})")
        return " AND ".join(parts) if parts else "all:*"


def _parse_published(entry) -> str | None:
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=UTC).isoformat()
