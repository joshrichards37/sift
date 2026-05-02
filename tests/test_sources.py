from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from sift.config import Preferences, SourcePref
from sift.sources import build_sources
from sift.sources.hn import HackerNewsSource
from sift.sources.reddit import RedditSource


# ── HackerNewsSource: query splitting ───────────────────────────────────────


async def test_hn_query_splits_on_or() -> None:
    """The Algolia HN search treats `OR` as a literal word, so we fan out
    parallel sub-queries client-side. Drift here = silently broken multi-keyword
    searches."""
    src = HackerNewsSource(id="hn", query="claude OR anthropic OR cursor")
    captured: list[str] = []

    async def fake_search(_client: Any, q: str) -> list:
        captured.append(q)
        return []

    src._search = fake_search  # type: ignore[method-assign]
    await src.poll()
    assert captured == ["claude", "anthropic", "cursor"]


async def test_hn_query_no_or_means_single_search() -> None:
    src = HackerNewsSource(id="hn", query="just-one-keyword")
    captured: list[str] = []

    async def fake_search(_client: Any, q: str) -> list:
        captured.append(q)
        return []

    src._search = fake_search  # type: ignore[method-assign]
    await src.poll()
    assert captured == ["just-one-keyword"]


async def test_hn_query_strips_whitespace_around_or() -> None:
    src = HackerNewsSource(id="hn", query="  a OR  b  OR c  ")
    captured: list[str] = []

    async def fake_search(_client: Any, q: str) -> list:
        captured.append(q)
        return []

    src._search = fake_search  # type: ignore[method-assign]
    await src.poll()
    assert captured == ["a", "b", "c"]


# ── RedditSource: payload parsing ───────────────────────────────────────────


def _reddit_payload(*items: dict[str, Any]) -> dict[str, Any]:
    return {"data": {"children": [{"data": d} for d in items]}}


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None: ...
    def json(self) -> dict: return self._payload


class _FakeAsyncClient:
    """Minimal stand-in for httpx.AsyncClient that returns a canned payload."""
    payload: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    async def __aenter__(self) -> "_FakeAsyncClient": return self
    async def __aexit__(self, *_: Any) -> None: ...
    async def get(self, _url: str, params: dict | None = None) -> _FakeResponse:
        return _FakeResponse(self.payload)


@pytest.fixture
def fake_reddit(monkeypatch: pytest.MonkeyPatch):
    """Patches httpx.AsyncClient to return whatever payload the test sets."""
    def _set(payload: dict) -> None:
        _FakeAsyncClient.payload = payload
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    return _set


async def test_reddit_filters_low_score(fake_reddit) -> None:
    fake_reddit(_reddit_payload(
        {"title": "low", "url": "https://x", "score": 5, "is_self": False, "stickied": False},
        {"title": "high", "url": "https://y", "score": 500, "is_self": False, "stickied": False,
         "created_utc": 1714600000, "author": "alice"},
    ))
    src = RedditSource(id="reddit:test", subreddit="programming", min_points=100)
    arts = await src.poll()
    assert len(arts) == 1
    assert arts[0].title == "high"


async def test_reddit_drops_stickied_posts(fake_reddit) -> None:
    """Stickied posts are mod announcements, not content — drop regardless of score."""
    fake_reddit(_reddit_payload(
        {"title": "mod sticky", "url": "https://x", "score": 9999,
         "is_self": False, "stickied": True},
    ))
    src = RedditSource(id="reddit:test", subreddit="programming", min_points=10)
    arts = await src.poll()
    assert arts == []


async def test_reddit_self_post_url_rewritten_to_permalink(fake_reddit) -> None:
    """Self-posts: link should point at the discussion permalink, not the
    self-URL (which is the same thing in a non-useful form)."""
    fake_reddit(_reddit_payload(
        {"title": "Discussion", "url": "https://reddit.com/r/x/y/", "score": 200,
         "is_self": True, "stickied": False, "selftext": "body text",
         "permalink": "/r/test/comments/abc/discussion/", "created_utc": 1714600000,
         "author": "op"},
    ))
    src = RedditSource(id="reddit:test", subreddit="test", min_points=50)
    arts = await src.poll()
    assert len(arts) == 1
    assert arts[0].url == "https://www.reddit.com/r/test/comments/abc/discussion/"
    assert arts[0].body == "body text"


async def test_reddit_link_post_keeps_external_url(fake_reddit) -> None:
    fake_reddit(_reddit_payload(
        {"title": "External link", "url_overridden_by_dest": "https://blog.example/post",
         "url": "https://reddit.com/...", "score": 200, "is_self": False, "stickied": False},
    ))
    src = RedditSource(id="reddit:test", subreddit="test", min_points=50)
    arts = await src.poll()
    assert len(arts) == 1
    assert arts[0].url == "https://blog.example/post"


async def test_reddit_handles_empty_payload(fake_reddit) -> None:
    fake_reddit({"data": {"children": []}})
    src = RedditSource(id="reddit:test", subreddit="test", min_points=50)
    arts = await src.poll()
    assert arts == []


# ── build_sources factory ───────────────────────────────────────────────────


def _prefs(*sources: SourcePref) -> Preferences:
    return Preferences(topics="test", sources=list(sources))


def test_factory_dispatches_each_kind() -> None:
    prefs = _prefs(
        SourcePref(id="rss:blog", url="https://example.com/feed", enabled=True),
        SourcePref(id="hn", query="x", enabled=True),
        SourcePref(id="reddit:programming", subreddit="programming", enabled=True),
    )
    sources = build_sources(prefs)
    assert len(sources) == 3
    kinds = {type(s).__name__ for s in sources}
    assert kinds == {"RSSSource", "HackerNewsSource", "RedditSource"}


def test_factory_skips_disabled_sources() -> None:
    prefs = _prefs(
        SourcePref(id="hn", query="x", enabled=False),
        SourcePref(id="reddit:programming", subreddit="programming", enabled=True),
    )
    sources = build_sources(prefs)
    assert len(sources) == 1
    assert isinstance(sources[0], RedditSource)


def test_factory_raises_for_reddit_without_subreddit() -> None:
    prefs = _prefs(SourcePref(id="reddit:bad", enabled=True))  # no subreddit set
    with pytest.raises(ValueError, match="reddit source.*requires 'subreddit'"):
        build_sources(prefs)


def test_factory_raises_for_unknown_kind() -> None:
    prefs = _prefs(SourcePref(id="mastodon:user", enabled=True))
    with pytest.raises(ValueError, match="unknown source kind"):
        build_sources(prefs)


# ── Suppress unused-import warning for _patch when not used ─────────────────
_ = patch  # noqa: F401  (kept available for future tests that need direct patching)
