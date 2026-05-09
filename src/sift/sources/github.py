from __future__ import annotations

import os

import httpx

from sift.sources.base import Article, Source


class GitHubReleasesSource(Source):
    """Tracks releases for one GitHub repository.

    `repo` is "owner/name" (e.g. "vllm-project/vllm"). Polls
    /repos/{repo}/releases and emits one Article per release. Drafts are
    always skipped; pre-releases are skipped unless `prereleases=True`.

    Authentication is optional: if GITHUB_TOKEN is set in the environment,
    we send it as a Bearer token (rate limit jumps from 60/h to 5000/h).
    For a personal bot polling a handful of repos hourly, anonymous is
    plenty.
    """

    def __init__(
        self,
        *,
        id: str,
        repo: str,
        cadence_seconds: int = 3600,
        prereleases: bool = False,
    ) -> None:
        self.id = id
        self.repo = repo
        self.cadence_seconds = cadence_seconds
        self.prereleases = prereleases

    async def poll(self) -> list[Article]:
        url = f"https://api.github.com/repos/{self.repo}/releases"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sift-bot/0.1",
        }
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            resp = await client.get(url, params={"per_page": 10})
            resp.raise_for_status()
            payload = resp.json()

        out: list[Article] = []
        for release in payload:
            if release.get("draft"):
                continue
            if release.get("prerelease") and not self.prereleases:
                continue
            html_url = release.get("html_url")
            tag = release.get("tag_name")
            if not html_url or not tag:
                continue
            # `name` is the human title (often empty or matches the tag); fall
            # back to the tag so the digest line always carries something useful.
            display = release.get("name") or tag
            title = f"{self.repo} {display}"
            body = (release.get("body") or "")[:8000]
            posted_at = release.get("published_at")
            author = (release.get("author") or {}).get("login")
            out.append(
                Article(
                    source_id=self.id,
                    url=html_url,
                    title=title,
                    body=body,
                    author=author,
                    posted_at=posted_at,
                )
            )
        return out
