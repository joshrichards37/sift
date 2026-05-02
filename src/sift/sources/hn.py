from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx

from sift.sources.base import Article, Source

ALGOLIA_URL = "https://hn.algolia.com/api/v1/search_by_date"


class HackerNewsSource(Source):
    """Hacker News via the Algolia search API. Filters by query + min points.

    The query string is split on " OR " into separate sub-queries that run in
    parallel; results are merged and deduplicated by URL. Algolia's HN search
    does not understand boolean OR operators inside a single query — it treats
    OR as a literal word — so we have to fan out client-side.
    """

    def __init__(
        self, *, id: str, query: str, min_points: int = 50, cadence_seconds: int = 1800
    ) -> None:
        self.id = id
        self.query = query
        self.min_points = min_points
        self.cadence_seconds = cadence_seconds

    async def poll(self) -> list[Article]:
        sub_queries = (
            [q.strip() for q in self.query.split(" OR ")] if " OR " in self.query else [self.query]
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            batches = await asyncio.gather(*(self._search(client, q) for q in sub_queries))

        seen: set[str] = set()
        out: list[Article] = []
        for batch in batches:
            for art in batch:
                if art.url in seen:
                    continue
                seen.add(art.url)
                out.append(art)
        return out

    async def _search(self, client: httpx.AsyncClient, query: str) -> list[Article]:
        params = {
            "query": query,
            "tags": "story",
            "numericFilters": f"points>={self.min_points}",
            "hitsPerPage": 30,
        }
        resp = await client.get(ALGOLIA_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

        out: list[Article] = []
        for hit in payload.get("hits", []):
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}"
            title = hit.get("title")
            if not title:
                continue
            posted_at = None
            ts = hit.get("created_at_i")
            if ts:
                posted_at = datetime.fromtimestamp(ts, tz=UTC).isoformat()
            out.append(
                Article(
                    source_id=self.id,
                    url=url,
                    title=title,
                    body=hit.get("story_text") or "",
                    author=hit.get("author"),
                    posted_at=posted_at,
                )
            )
        return out
