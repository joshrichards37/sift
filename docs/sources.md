# Sources

Sources are the things sift polls. Every source has a stable string id (e.g. `hn`, `rss:simonwillison`, `bsky:simonw`) and a `cadence_seconds` controlling how often it's polled. The id's prefix before the colon picks the implementation:

| Prefix | Class | What it does |
|---|---|---|
| `hn` | `HackerNewsSource` | Algolia search API. `query` is free-text; `min_points` filters to popular stories. |
| `rss:` | `RSSSource` | Anything `feedparser` handles — RSS, Atom, YouTube channel feeds, Substack feeds. |
| `reddit:` | `RedditSource` | One subreddit (or `+`-joined combo) via public JSON API. `min_points` filters by score. |
| `bsky:` | `BlueskySource` | One handle's author feed. Needs `BLUESKY_HANDLE` and `BLUESKY_APP_PASSWORD` in `.env`. |

## Configuring sources

Sources are listed in `preferences.yaml`. Each has the same shape:

```yaml
sources:
  - id: rss:hamel
    enabled: true
    url: https://hamel.dev/index.xml
    cadence_seconds: 14400
```

Set `enabled: false` to leave a source defined but skipped — useful for keeping config history without deleting URLs.

### Hacker News quirk: no boolean OR in the query

Algolia's HN search doesn't support boolean `OR` inside a single query string — it treats `OR` as a literal word. The source splits on `" OR "` and fans out parallel sub-queries client-side, merging by URL. So this works:

```yaml
- id: hn
  query: "Claude OR Anthropic OR Cursor OR \"AI agent\""
```

But `(Claude AND Anthropic)` does not — that's a single query in Algolia and behaves as a phrase match.

### Reddit

Use the dedicated `reddit:` source — it hits the public JSON API (`/r/<sub>/top.json?t=day`) and lets you filter by post score *before* burning an LLM call. The plain `/r/<sub>/.rss` route through the `rss:` source still works, but it gives you no min-score knob and is noisier.

```yaml
- id: reddit:programming
  enabled: true
  subreddit: programming           # single sub
  min_points: 200
  cadence_seconds: 3600

- id: reddit:tech-roundup
  enabled: true
  subreddit: programming+rust+golang+webdev   # combined feed (Reddit-native + syntax)
  min_points: 100
  cadence_seconds: 3600
```

The source uses a non-default User-Agent (`sift-bot/0.1 …`) to avoid the generic-UA block. No auth required. Reddit rate-limits aggressively at high cadences — keep `cadence_seconds` ≥ 1800 per source.

Self-posts (text posts) get the `selftext` field as the article body, capped at 4000 chars. Link posts have empty bodies; the LLM scores against title alone.

### YouTube channels

YouTube exposes a free Atom feed per channel at `https://www.youtube.com/feeds/videos.xml?channel_id=<UC…>`. The feed has the latest 15 videos; title = video title, body = description.

Modern channel URLs use `@handle` (e.g. `https://www.youtube.com/@AndrejKarpathy`) but the feed wants the legacy `UC…` channel ID. Two ways to resolve:

**Manual (fastest):** open the channel page in a browser → View Page Source (`Ctrl+U` / `⌘U`) → `Ctrl+F` for `<link rel="canonical"`. The href contains `/channel/UC…` — that's the channel ID.

**Programmatic:** YouTube's GDPR consent gate strips the page contents unless the right cookies are set. Use both `CONSENT=YES+` and `SOCS=CAI`:

```python
import httpx, re
r = httpx.get(
    "https://www.youtube.com/@AndrejKarpathy",
    headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
    cookies={"CONSENT": "YES+", "SOCS": "CAI"},
    follow_redirects=True,
)
m = re.search(r'<link rel="canonical" href="https://www\.youtube\.com/channel/(UC[\w-]+)"', r.text)
print(m.group(1) if m else "not found")
# UCXUPKJO5MZQN11PqgIvyuvQ
```

Drop the resolved ID into `preferences.yaml` as a normal `rss:` source — no dedicated `youtube:` source needed:

```yaml
- id: rss:karpathy-youtube
  enabled: true
  url: https://www.youtube.com/feeds/videos.xml?channel_id=UCXUPKJO5MZQN11PqgIvyuvQ
  cadence_seconds: 21600   # YT channels post slowly; 6h is plenty
```

The `ai-tooling` preset ships a real example — see [`examples/preferences-ai-tooling.yaml`](../examples/preferences-ai-tooling.yaml).

### Bluesky

`BlueskySource` reads one handle's author feed (not the firehose). The post text becomes the article body; the URL is the first link in the post if any, otherwise the post permalink. Login is via app password ([create one](https://bsky.app/settings/app-passwords) — never your main password).

### Twitter / X — not supported, here's why

There is no `twitter:` / `x:` source by design. Every available path has a fatal flaw for a self-hosted personal bot:

- **Official X API.** Free tier reads ~1500 tweets/month total — useless for following anyone active. Basic tier ($100/mo) reads ~10k tweets/month — still thin and the wrong shape. Pro tier ($5k/mo) is enterprise-priced, not personal-bot-priced.
- **Nitter / third-party RSS frontends.** Public Nitter instances have been largely dead since 2024 as X rotated auth and ratelimited the guest-token endpoints they depended on. Self-hosting Nitter is possible but breaks every few weeks; the operational pain isn't worth it for a hobby.
- **Direct scraping.** Against X's ToS and gets logged-out accounts blocked within hours. Don't.

**The recommended substitute is Bluesky.** Many of the AI / engineering accounts you'd want from X (Karpathy, Simon Willison, Hamel, etc.) crosspost there or have moved over entirely; the `bsky:` source covers them with no auth pain. For people who only post on X, big threads usually surface on Hacker News within hours — the `hn` source catches them by proxy.

If you absolutely must have an X-only account in the feed, your only real option is to run a self-hosted Nitter instance and point an `rss:` source at its per-user feed. That's a maintenance commitment, not a setup step.

| Source velocity | Suggested cadence |
|---|---|
| Aggregator (HN front page, big subs) | 1800 (30 min) |
| Newsletter / news outlet | 3600 (1 hr) |
| Personal blog | 14400 (4 hr) |
| Slow-poster (Karpathy, Sam Altman, Andy Matuschak) | 21600 (6 hr) |
| arXiv RSS | 21600 (6 hr) |

Faster cadences mean more LLM scoring work; the `max_per_cycle` cap prevents runaway GPU usage when a backlog drains.

## Adding a new source

1. New file `src/sift/sources/<name>.py` subclassing `Source`. Implement `async def poll() -> list[Article]`. Idempotent — caller dedups by URL.

   ```python
   from sift.sources.base import Article, Source

   class MyMastoSource(Source):
       def __init__(self, *, id: str, instance: str, account: str, cadence_seconds: int):
           self.id = id
           self.instance = instance
           self.account = account
           self.cadence_seconds = cadence_seconds

       async def poll(self) -> list[Article]:
           # fetch posts, return Article instances
           ...
   ```

2. Register in `src/sift/sources/__init__.py`'s `build_sources` factory under a new id-prefix branch:

   ```python
   elif kind == "masto":
       out.append(MyMastoSource(id=s.id, instance=s.url, account=s.handle, cadence_seconds=s.cadence_seconds))
   ```

3. Add the relevant fields to `SourcePref` in `config.py` if you need new ones (e.g. `account: str | None`).

4. Document in `preferences.example.yaml` under `sources:` with the default cadence.

## Why not just use one mega-feed aggregator?

Could feed in a polished aggregator (e.g. point at Feedly's per-collection RSS) and skip per-source code. Two reasons we don't:

1. **Per-source cadence**: arXiv polls at 6h, Latent Space at 1h, HN at 30 min. One aggregator gives you the slowest refresh.
2. **Per-source rate limiting + backoff**: the scheduler isolates failures — if Reddit 429s, only that loop sleeps; everything else keeps running.

Per-source plugins are ~30 lines each. The complexity is worth it.
