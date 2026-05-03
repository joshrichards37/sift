from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from html import escape
from typing import Protocol

from sift.config import Preferences, Settings
from sift.recommender import record_for_chat, suggest_for_chat
from sift.storage import (
    connect,
    count_unpushed,
    fetch_top_unpushed,
    mark_many_pushed,
)

log = logging.getLogger(__name__)


DigestItem = tuple[int, str]  # mirrors telegram_bot.DigestItem; kept duplicate to
# avoid digest.py importing the telegram package.


class SendText(Protocol):
    """Broadcast callable. `items` is optional — when provided, the underlying
    bot attaches a per-article thumbs keyboard to the final chunk of the message."""

    async def __call__(self, text: str, *, items: list[DigestItem] | None = None) -> None: ...


async def digest_loop(settings: Settings, prefs: Preferences, send: SendText) -> None:
    """Sleeps until the next digest_time, then sends; repeats forever."""
    while True:
        next_run = _next_digest_time(prefs.digest_time)
        sleep_s = max(1.0, (next_run - datetime.now()).total_seconds())
        log.info("next digest at %s (in %.0f min)", next_run.isoformat(), sleep_s / 60)
        await asyncio.sleep(sleep_s)
        try:
            populate_suggestions(settings, prefs)
            await run_digest(settings, prefs, send)
        except Exception:
            log.exception("digest failed")


def populate_suggestions(settings: Settings, prefs: Preferences) -> int:
    """Run the taste-discovery suggester for each authorised chat and record
    any qualifying suggestion. Returns the count recorded.

    Idempotent at the per-chat level — if a chat already has a pending
    (un-responded) suggestion, the digest UX layer surfaces that one and
    the recommender is free to add another. We don't dedupe here because
    the storage layer enforces 'one pending shown at a time' via the query
    in pending_suggestion_for, and the per-chat mute/decline checks in the
    suggester prevent re-suggesting topics the user has already rejected."""
    recorded = 0
    with connect(settings.db_path) as conn:
        for chat_id in settings.chat_ids:
            suggestion = suggest_for_chat(conn, chat_id=str(chat_id), prefs_topics=prefs.topics)
            if suggestion is None:
                continue
            record_for_chat(conn, chat_id=str(chat_id), suggestion=suggestion)
            log.info(
                "recorded suggestion for chat %s: topic=%r confidence=%.2f",
                chat_id,
                suggestion.topic,
                suggestion.confidence,
            )
            recorded += 1
    return recorded


async def run_digest(settings: Settings, prefs: Preferences, send: SendText) -> int:
    """Send the daily digest. Returns the number of articles included (0 if quiet)."""
    with connect(settings.db_path) as conn:
        rows = fetch_top_unpushed(
            conn,
            min_score=prefs.relevance_threshold,
            limit=prefs.digest_size,
        )
    if not rows:
        log.info("digest: nothing to send")
        await send(_no_news_message())
        return 0
    msg = format_digest(rows, header="📰 Daily Digest")
    items = _build_thumbs_items(rows)
    await send(msg, items=items)
    with connect(settings.db_path) as conn:
        mark_many_pushed(conn, [r["id"] for r in rows])
    log.info("digest: sent %d articles", len(rows))
    return len(rows)


async def run_more(settings: Settings, prefs: Preferences, send: SendText, n: int) -> int:
    """Send the next N unpushed articles (above threshold). Returns count sent."""
    with connect(settings.db_path) as conn:
        rows = fetch_top_unpushed(conn, min_score=prefs.relevance_threshold, limit=n)
    if not rows:
        await send("Nothing left in the backlog above threshold. Wait for the next digest.")
        return 0
    msg = format_digest(rows, header=f"More from backlog ({len(rows)})")
    items = _build_thumbs_items(rows)
    await send(msg, items=items)
    with connect(settings.db_path) as conn:
        mark_many_pushed(conn, [r["id"] for r in rows])
    return len(rows)


def _build_thumbs_items(rows: list[sqlite3.Row]) -> list[DigestItem]:
    """Pair each digest item's visible 1-based index with its article id so the
    bot can build the thumbs keyboard. Numbering matches format_digest's
    enumerate(rows, 1)."""
    return [(i, r["id"]) for i, r in enumerate(rows, 1)]


def get_backlog_count(settings: Settings, prefs: Preferences) -> int:
    with connect(settings.db_path) as conn:
        return count_unpushed(conn, prefs.relevance_threshold)


def format_digest(rows: list[sqlite3.Row], *, header: str) -> str:
    today = datetime.now().strftime("%a %d %b")
    parts = [f"<b>{escape(header)} — {today}</b>", f"{len(rows)} top items.\n"]
    for i, r in enumerate(rows, 1):
        title = escape(r["title"])
        url = escape(r["url"], quote=True)
        source = escape(r["source_id"])
        summary = escape(r["summary"] or "")
        parts.append(
            f'<b>{i}. <a href="{url}">{title}</a></b>\n'
            f"<i>{source} · {r['relevance_score']}/10</i>\n"
            f"{summary}\n"
        )
    return "\n".join(parts)


def _no_news_message() -> str:
    return "📰 Daily Digest — nothing scored above threshold today. /backlog to peek."


def _next_digest_time(hhmm: str) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    now = datetime.now()
    today_run = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return today_run if today_run > now else today_run + timedelta(days=1)
