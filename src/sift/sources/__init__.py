from sift.sources.base import Article, Source

# Single source of truth for valid source-id prefixes. Imported by
# sift.config.SourcePref to validate ids at parse time so a typo like
# 'hn-finance' fails at load_preferences() instead of startup.
KNOWN_KINDS: frozenset[str] = frozenset({"hn", "rss", "reddit", "bsky"})

__all__ = ["Article", "KNOWN_KINDS", "Source", "build_sources"]


def build_sources(prefs):
    from sift.sources.bluesky import BlueskySource
    from sift.sources.hn import HackerNewsSource
    from sift.sources.reddit import RedditSource
    from sift.sources.rss import RSSSource

    out: list[Source] = []
    for s in prefs.sources:
        if not s.enabled:
            continue
        kind = s.id.split(":", 1)[0]
        if kind == "rss":
            out.append(RSSSource(id=s.id, url=s.url, cadence_seconds=s.cadence_seconds))
        elif kind == "hn":
            out.append(
                HackerNewsSource(
                    id=s.id,
                    query=s.query or "",
                    min_points=s.min_points or 50,
                    cadence_seconds=s.cadence_seconds,
                )
            )
        elif kind == "reddit":
            if not s.subreddit:
                raise ValueError(f"reddit source {s.id!r} requires 'subreddit' field")
            out.append(
                RedditSource(
                    id=s.id,
                    subreddit=s.subreddit,
                    min_points=s.min_points or 50,
                    cadence_seconds=s.cadence_seconds,
                )
            )
        elif kind == "bsky":
            out.append(BlueskySource(id=s.id, handle=s.handle, cadence_seconds=s.cadence_seconds))
        else:
            raise ValueError(f"unknown source kind: {s.id!r}")
    return out
