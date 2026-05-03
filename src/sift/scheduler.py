from __future__ import annotations

import asyncio
import logging
import random

from sift.config import Preferences, Settings
from sift.llm import LLM
from sift.sources.base import Article, Source
from sift.storage import connect, insert_article, mark_scored

log = logging.getLogger(__name__)


async def run_scheduler(
    sources: list[Source], settings: Settings, prefs: Preferences, llm: LLM
) -> None:
    """Run all source poll loops in parallel. Articles are scored + summarised
    and stashed in the DB; the digest job (digest.py) picks them up later."""
    await asyncio.gather(*(_poll_loop(s, settings, prefs, llm) for s in sources))


async def _poll_loop(source: Source, settings: Settings, prefs: Preferences, llm: LLM) -> None:
    while True:
        try:
            await _poll_once(source, settings, prefs, llm)
        except Exception:
            log.exception("source %s: poll failed", source.id)
        sleep_s = source.cadence_seconds * random.uniform(0.9, 1.1)
        await asyncio.sleep(sleep_s)


async def _poll_once(source: Source, settings: Settings, prefs: Preferences, llm: LLM) -> None:
    items = await source.poll()
    log.info("source %s: fetched %d items", source.id, len(items))

    new_articles: list[tuple[str, Article]] = []
    with connect(settings.db_path) as conn:
        for art in items:
            aid = insert_article(
                conn,
                source_id=art.source_id,
                url=art.url,
                title=art.title,
                body=art.body,
                author=art.author,
                posted_at=art.posted_at,
            )
            if aid:
                new_articles.append((aid, art))

    if not new_articles:
        return

    for scored, (aid, art) in enumerate(new_articles):
        if scored >= prefs.max_per_cycle:
            log.info(
                "source %s: hit max_per_cycle (%d), deferring %d items",
                source.id,
                prefs.max_per_cycle,
                len(new_articles) - scored,
            )
            break
        score = await llm.score_relevance(art, prefs)
        summary: str | None = None
        if score.score >= prefs.relevance_threshold:
            summary = await llm.summarize(art, prefs)
        with connect(settings.db_path) as conn:
            mark_scored(conn, aid, score.score, summary, topic_tags=score.topic_tags)
