from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import feedparser

from sift.sources.base import Article, Source


class RSSSource(Source):
    def __init__(self, *, id: str, url: str, cadence_seconds: int = 1800) -> None:
        self.id = id
        self.url = url
        self.cadence_seconds = cadence_seconds

    async def poll(self) -> list[Article]:
        # feedparser is sync + blocking; run in a thread to keep the loop free.
        feed = await asyncio.to_thread(feedparser.parse, self.url)
        out: list[Article] = []
        for entry in feed.entries:
            url = getattr(entry, "link", None)
            title = getattr(entry, "title", None)
            if not url or not title:
                continue
            body = getattr(entry, "summary", None) or getattr(entry, "description", None) or ""
            out.append(
                Article(
                    source_id=self.id,
                    url=url,
                    title=title,
                    body=body,
                    author=getattr(entry, "author", None),
                    posted_at=_parse_published(entry),
                )
            )
        return out


def _parse_published(entry) -> str | None:
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=UTC).isoformat()
