from __future__ import annotations

import html as html_module
import re

import httpx

from sift.sources.base import Article, Source

# Mastodon handle: "user@instance.tld". The leading @ is optional in
# preferences.yaml — we strip it before parsing.
HANDLE_RE = re.compile(r"^([^@\s]+)@([^@\s]+)$")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LINK_RE = re.compile(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', flags=re.DOTALL)


class MastodonSource(Source):
    """Public posts from one Mastodon account.

    `handle` is the federated identifier "user@instance.tld" (e.g.
    "simon@simonwillison.net"). No auth needed for public posts —
    Mastodon's /api/v1/accounts/* endpoints are anonymous-friendly.
    Resolves the handle to an account id once via /lookup, then polls
    statuses on each cycle. Reblogs (boosts) are unwrapped to the
    original status so the LLM scores the boosted content, not "X
    boosted Y."
    """

    def __init__(
        self,
        *,
        id: str,
        handle: str,
        cadence_seconds: int = 1800,
        limit: int = 20,
    ) -> None:
        self.id = id
        self.handle = handle
        self.cadence_seconds = cadence_seconds
        self.limit = limit
        self._account_id: str | None = None
        m = HANDLE_RE.match((handle or "").strip().lstrip("@"))
        if not m:
            self.disabled = True
            self.disabled_reason = f"invalid Mastodon handle (need 'user@instance.tld'): {handle!r}"
            self._username = ""
            self._instance = ""
        else:
            self._username, self._instance = m.group(1), m.group(2)

    async def poll(self) -> list[Article]:
        if self.disabled:
            return []
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"Accept": "application/json", "User-Agent": "sift-bot/0.1"},
        ) as client:
            account_id = await self._ensure_account_id(client)
            if account_id is None:
                return []
            url = f"https://{self._instance}/api/v1/accounts/{account_id}/statuses"
            resp = await client.get(url, params={"limit": self.limit, "exclude_replies": "true"})
            if resp.status_code == 404:
                self.disabled = True
                self.disabled_reason = "statuses endpoint 404 — account may be suspended"
                return []
            resp.raise_for_status()
            statuses = resp.json()

        out: list[Article] = []
        for status in statuses:
            origin = status.get("reblog") or status
            text = _strip_html(origin.get("content") or "")
            permalink = origin.get("url")
            if not permalink or not text:
                continue
            account = origin.get("account") or {}
            author = account.get("acct") or account.get("username")
            out.append(
                Article(
                    source_id=self.id,
                    url=permalink,
                    title=text.split("\n", 1)[0][:140],
                    body=text,
                    author=author,
                    posted_at=origin.get("created_at"),
                )
            )
        return out

    async def _ensure_account_id(self, client: httpx.AsyncClient) -> str | None:
        """Resolve the handle to a numeric account id once. Failures here are
        treated as transient (instance outage, rate limit) and do NOT disable
        the source — only a 404 from /lookup is permanent."""
        if self._account_id is not None:
            return self._account_id
        lookup = f"https://{self._instance}/api/v1/accounts/lookup"
        try:
            r = await client.get(lookup, params={"acct": self._username})
        except Exception:
            return None
        if r.status_code == 404:
            self.disabled = True
            self.disabled_reason = f"account @{self._username}@{self._instance} not found"
            return None
        if r.status_code >= 400:
            return None
        try:
            account_id = r.json().get("id")
        except Exception:
            return None
        if not account_id:
            return None
        self._account_id = str(account_id)
        return self._account_id


def _strip_html(html: str) -> str:
    """Mastodon serves post bodies as HTML. The LLM scorer would survive raw
    HTML but it bloats the token budget — strip to plain text, preserving
    link targets so URL signal isn't lost."""
    text = _LINK_RE.sub(r"\2 (\1)", html)
    text = text.replace("</p>", "\n\n").replace("<br>", "\n").replace("<br />", "\n")
    text = _HTML_TAG_RE.sub("", text)
    return html_module.unescape(text).strip()
