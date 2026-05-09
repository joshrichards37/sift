from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from sift.config import Preferences, SourcePref
from sift.sources import build_sources
from sift.sources.github import GitHubReleasesSource
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
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None: ...
    def json(self) -> dict: return self._payload


class _FakeAsyncClient:
    """Minimal stand-in for httpx.AsyncClient that returns a canned payload."""
    payload: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    async def __aenter__(self) -> _FakeAsyncClient: return self
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


# ── Source self-disable on permanent failure ───────────────────────────────


class _FakeStatusResponse:
    """Reddit /top.json response with a configurable status code. Used to
    simulate 404/403 without going to the network."""

    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload or {"data": {"children": []}}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)  # type: ignore[arg-type]

    def json(self) -> dict:
        return self._payload


class _FakeStatusClient:
    status: int = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    async def __aenter__(self) -> _FakeStatusClient:
        return self

    async def __aexit__(self, *_: Any) -> None: ...

    async def get(self, _url: str, params: dict | None = None) -> _FakeStatusResponse:
        return _FakeStatusResponse(_FakeStatusClient.status)


@pytest.fixture
def fake_reddit_status(monkeypatch: pytest.MonkeyPatch):
    def _set(status: int) -> None:
        _FakeStatusClient.status = status
        monkeypatch.setattr(httpx, "AsyncClient", _FakeStatusClient)

    return _set


async def test_reddit_disables_self_on_404(fake_reddit_status) -> None:
    """A 404 from /top.json means the subreddit doesn't exist. The source
    should mark itself disabled so the scheduler stops polling it; otherwise
    we'd burn cadence cycles forever on an LLM-hallucinated sub name."""
    fake_reddit_status(404)
    src = RedditSource(id="reddit:nope", subreddit="nope", min_points=50)
    assert src.disabled is False
    arts = await src.poll()
    assert arts == []
    assert src.disabled is True
    assert "404" in src.disabled_reason


async def test_reddit_disables_self_on_403(fake_reddit_status) -> None:
    fake_reddit_status(403)
    src = RedditSource(id="reddit:private", subreddit="private", min_points=50)
    arts = await src.poll()
    assert arts == []
    assert src.disabled is True
    assert "403" in src.disabled_reason


async def test_reddit_disabled_source_returns_empty_without_request(monkeypatch) -> None:
    """A pre-disabled source should never hit the network. Patch the client
    to a sentinel that raises if instantiated."""

    def _explode(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("disabled source must not make HTTP requests")

    monkeypatch.setattr(httpx, "AsyncClient", _explode)
    src = RedditSource(id="reddit:dead", subreddit="dead", min_points=50)
    src.disabled = True
    src.disabled_reason = "test"
    arts = await src.poll()
    assert arts == []


def test_bluesky_disables_self_when_creds_missing(monkeypatch) -> None:
    """Construction-time check: missing BLUESKY_* should disable the source
    immediately so the scheduler never even starts its poll loop."""
    from sift.sources.bluesky import BlueskySource

    monkeypatch.delenv("BLUESKY_HANDLE", raising=False)
    monkeypatch.delenv("BLUESKY_APP_PASSWORD", raising=False)
    src = BlueskySource(id="bsky:test", handle="example.bsky.social")
    assert src.disabled is True
    assert "BLUESKY_HANDLE" in src.disabled_reason


def test_bluesky_not_disabled_when_creds_present(monkeypatch) -> None:
    from sift.sources.bluesky import BlueskySource

    monkeypatch.setenv("BLUESKY_HANDLE", "test.bsky.social")
    monkeypatch.setenv("BLUESKY_APP_PASSWORD", "test-password")
    src = BlueskySource(id="bsky:test", handle="example.bsky.social")
    assert src.disabled is False


# ── GitHubReleasesSource: payload parsing + filtering ─────────────────────


def _gh_release(
    *,
    tag: str,
    name: str | None = None,
    body: str = "",
    draft: bool = False,
    prerelease: bool = False,
    author: str = "octocat",
    published: str = "2026-05-01T12:00:00Z",
) -> dict[str, Any]:
    return {
        "tag_name": tag,
        "name": name,
        "body": body,
        "draft": draft,
        "prerelease": prerelease,
        "author": {"login": author},
        "published_at": published,
        "html_url": f"https://github.com/example/repo/releases/tag/{tag}",
    }


class _FakeReleasesResponse:
    def __init__(self, payload: list[dict]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None: ...
    def json(self) -> list[dict]:
        return self._payload


class _FakeReleasesClient:
    """Minimal AsyncClient stand-in that captures the request headers and
    returns canned releases. Used to assert auth-header behavior."""

    payload: list[dict] = []
    captured_headers: dict[str, str] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _FakeReleasesClient.captured_headers = dict(kwargs.get("headers") or {})

    async def __aenter__(self) -> _FakeReleasesClient:
        return self

    async def __aexit__(self, *_: Any) -> None: ...

    async def get(self, _url: str, params: dict | None = None) -> _FakeReleasesResponse:
        return _FakeReleasesResponse(self.payload)


@pytest.fixture
def fake_github(monkeypatch: pytest.MonkeyPatch):
    def _set(payload: list[dict]) -> None:
        _FakeReleasesClient.payload = payload
        _FakeReleasesClient.captured_headers = None
        monkeypatch.setattr(httpx, "AsyncClient", _FakeReleasesClient)

    return _set


async def test_github_skips_drafts(fake_github) -> None:
    fake_github(
        [
            _gh_release(tag="v1.0.0", name="First stable"),
            _gh_release(tag="v1.1.0-draft", draft=True),
        ]
    )
    src = GitHubReleasesSource(id="github:test", repo="example/repo")
    arts = await src.poll()
    assert len(arts) == 1
    assert arts[0].title.endswith("First stable")


async def test_github_skips_prereleases_by_default(fake_github) -> None:
    fake_github(
        [
            _gh_release(tag="v1.0.0"),
            _gh_release(tag="v1.1.0-rc1", prerelease=True),
        ]
    )
    src = GitHubReleasesSource(id="github:test", repo="example/repo")
    arts = await src.poll()
    assert {a.title.split()[-1] for a in arts} == {"v1.0.0"}


async def test_github_includes_prereleases_when_opted_in(fake_github) -> None:
    fake_github(
        [
            _gh_release(tag="v1.0.0"),
            _gh_release(tag="v1.1.0-rc1", prerelease=True),
        ]
    )
    src = GitHubReleasesSource(id="github:test", repo="example/repo", prereleases=True)
    arts = await src.poll()
    assert len(arts) == 2


async def test_github_falls_back_to_tag_when_name_missing(fake_github) -> None:
    """Many repos leave `name` empty and only set `tag_name`. Title should
    still carry the version so the digest line is informative."""
    fake_github([_gh_release(tag="v2.5.0", name=None)])
    src = GitHubReleasesSource(id="github:vllm", repo="vllm-project/vllm")
    arts = await src.poll()
    assert arts[0].title == "vllm-project/vllm v2.5.0"


async def test_github_uses_token_when_set(fake_github, monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    fake_github([_gh_release(tag="v1.0.0")])
    src = GitHubReleasesSource(id="github:test", repo="example/repo")
    await src.poll()
    headers = _FakeReleasesClient.captured_headers or {}
    assert headers.get("Authorization") == "Bearer ghp_secret"


async def test_github_anonymous_when_no_token(fake_github, monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    fake_github([_gh_release(tag="v1.0.0")])
    src = GitHubReleasesSource(id="github:test", repo="example/repo")
    await src.poll()
    headers = _FakeReleasesClient.captured_headers or {}
    assert "Authorization" not in headers


# ── build_sources factory ───────────────────────────────────────────────────


def _prefs(*sources: SourcePref) -> Preferences:
    return Preferences(topics="test", sources=list(sources))


def test_factory_dispatches_each_kind() -> None:
    prefs = _prefs(
        SourcePref(id="rss:blog", url="https://example.com/feed", enabled=True),
        SourcePref(id="hn", query="x", enabled=True),
        SourcePref(id="reddit:programming", subreddit="programming", enabled=True),
        SourcePref(id="github:vllm", repo="vllm-project/vllm", enabled=True),
    )
    sources = build_sources(prefs)
    assert len(sources) == 4
    kinds = {type(s).__name__ for s in sources}
    assert kinds == {"RSSSource", "HackerNewsSource", "RedditSource", "GitHubReleasesSource"}


def test_factory_raises_for_github_without_repo() -> None:
    prefs = _prefs(SourcePref(id="github:bad", enabled=True))  # no repo set
    with pytest.raises(ValueError, match="github source.*requires 'repo'"):
        build_sources(prefs)


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


def test_unknown_kind_rejected_at_pref_construction() -> None:
    """Validation moved up to SourcePref so the typo fails at preferences-load
    time rather than at build_sources. The test still proves the same rule —
    invalid kinds can't reach the factory."""
    with pytest.raises(ValueError, match="unknown source kind"):
        SourcePref(id="mastodon:user", enabled=True)


# ── Suppress unused-import warning for _patch when not used ─────────────────
_ = patch  # noqa: F401  (kept available for future tests that need direct patching)
