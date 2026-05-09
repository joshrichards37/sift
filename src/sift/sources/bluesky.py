from __future__ import annotations

import asyncio
import os
import re

from atproto import Client

from sift.sources.base import Article, Source

URL_RE = re.compile(r"https?://\S+")


class BlueskySource(Source):
    """Pulls recent posts from a single handle's feed. Article URL is the post permalink;
    body is the post text. Useful for tracking high-signal accounts (simonw, etc.)."""

    def __init__(self, *, id: str, handle: str, cadence_seconds: int = 900) -> None:
        self.id = id
        self.handle = handle
        self.cadence_seconds = cadence_seconds
        self._client: Client | None = None
        # Detect missing credentials at construction time so the scheduler
        # never starts the poll loop, rather than raising every cadence cycle.
        if not (os.environ.get("BLUESKY_HANDLE") and os.environ.get("BLUESKY_APP_PASSWORD")):
            self.disabled = True
            self.disabled_reason = "BLUESKY_HANDLE/BLUESKY_APP_PASSWORD not set in environment"

    def _ensure_client(self) -> Client:
        if self._client is not None:
            return self._client
        bsky_handle = os.environ.get("BLUESKY_HANDLE")
        bsky_pw = os.environ.get("BLUESKY_APP_PASSWORD")
        # Defensive — __init__ should have caught this and disabled the source,
        # but keep the explicit error for callers that bypass the scheduler.
        if not (bsky_handle and bsky_pw):
            raise RuntimeError(
                "BLUESKY_HANDLE and BLUESKY_APP_PASSWORD must be set to use bsky sources"
            )
        client = Client()
        client.login(bsky_handle, bsky_pw)
        self._client = client
        return client

    async def poll(self) -> list[Article]:
        if self.disabled:
            return []
        return await asyncio.to_thread(self._poll_sync)

    def _poll_sync(self) -> list[Article]:
        client = self._ensure_client()
        feed = client.get_author_feed(actor=self.handle, limit=30)
        out: list[Article] = []
        for item in feed.feed:
            post = item.post
            if not post or not post.record:
                continue
            text = getattr(post.record, "text", "") or ""
            # Prefer the first link in the post if any; otherwise the post URL itself.
            url = _first_url(text) or _bsky_post_url(post.uri, self.handle)
            if not url:
                continue
            out.append(
                Article(
                    source_id=self.id,
                    url=url,
                    title=text.split("\n", 1)[0][:140] or f"@{self.handle} post",
                    body=text,
                    author=self.handle,
                    posted_at=getattr(post.record, "created_at", None),
                )
            )
        return out


def _first_url(text: str) -> str | None:
    m = URL_RE.search(text)
    return m.group(0) if m else None


def _bsky_post_url(uri: str, handle: str) -> str | None:
    # at://did:plc:xxx/app.bsky.feed.post/<rkey> → https://bsky.app/profile/<handle>/post/<rkey>
    parts = uri.rsplit("/", 1)
    if len(parts) != 2:
        return None
    rkey = parts[1]
    return f"https://bsky.app/profile/{handle}/post/{rkey}"
