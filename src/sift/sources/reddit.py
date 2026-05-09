from __future__ import annotations

from datetime import UTC, datetime

import httpx

from sift.sources.base import Article, Source

# Reddit blocks generic UAs (`python-requests/x`, default httpx). Use anything
# unique. Format follows Reddit's loose recommendation; no auth needed for the
# public `top.json` endpoint, just don't look like a bot mass-scraper.
USER_AGENT = "sift-bot/0.1 (self-hosted personal news agent)"


class RedditSource(Source):
    """One subreddit (or `+`-joined combo) via the public JSON API.

    `subreddit` accepts a single name (`programming`) or a Reddit-native
    combo (`programming+rust+golang`). `min_points` filters out low-score
    posts before they reach the LLM scorer — set to 50+ on big subs.
    """

    def __init__(
        self,
        *,
        id: str,
        subreddit: str,
        min_points: int = 50,
        cadence_seconds: int = 3600,
    ) -> None:
        self.id = id
        self.subreddit = subreddit
        self.min_points = min_points
        self.cadence_seconds = cadence_seconds

    async def poll(self) -> list[Article]:
        if self.disabled:
            return []
        url = f"https://www.reddit.com/r/{self.subreddit}/top.json"
        params = {"t": "day", "limit": 50}
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            resp = await client.get(url, params=params)
            # 404 = subreddit doesn't exist; 403 = private/quarantined. Both
            # are permanent — disable the source instead of retrying every
            # cadence cycle until the user notices the log spam.
            if resp.status_code == 404:
                self.disabled = True
                self.disabled_reason = f"r/{self.subreddit} not found (404)"
                return []
            if resp.status_code == 403:
                self.disabled = True
                self.disabled_reason = f"r/{self.subreddit} is private or quarantined (403)"
                return []
            resp.raise_for_status()
            payload = resp.json()

        out: list[Article] = []
        for child in payload.get("data", {}).get("children", []):
            d = child.get("data", {})
            if d.get("stickied"):
                continue
            if d.get("score", 0) < self.min_points:
                continue
            title = d.get("title")
            if not title:
                continue
            link = d.get("url_overridden_by_dest") or d.get("url")
            if d.get("is_self"):
                # self-post: link to the discussion, not the self-URL
                link = "https://www.reddit.com" + d.get("permalink", "")
            if not link:
                continue
            body = (d.get("selftext") or "")[:4000]
            posted_at = None
            ts = d.get("created_utc")
            if ts:
                posted_at = datetime.fromtimestamp(ts, tz=UTC).isoformat()
            out.append(
                Article(
                    source_id=self.id,
                    url=link,
                    title=title,
                    body=body,
                    author=d.get("author"),
                    posted_at=posted_at,
                )
            )
        return out
