from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from typing import Protocol

from sift.adjacency import suggest_for_chat_via_adjacency
from sift.config import Preferences, Settings
from sift.llm import LLM
from sift.recommender import record_for_chat, suggest_for_chat
from sift.storage import (
    connect,
    count_unpushed,
    fetch_top_unpushed,
    mark_many_pushed,
    pending_suggestion_for,
)

log = logging.getLogger(__name__)


DigestItem = tuple[int, str]  # mirrors telegram_bot.DigestItem; kept duplicate to
# avoid digest.py importing the telegram package.


@dataclass(frozen=True, slots=True)
class SuggestionFooter:
    """Per-chat snapshot of a pending suggestion. The bot uses these fields to
    build the suggestion follow-up message after broadcasting the digest."""

    suggestion_id: int
    topic: str
    evidence_count: int


class SendText(Protocol):
    """Broadcast callable. `items` is optional — when provided, the underlying
    bot attaches a per-article thumbs keyboard to the final chunk of the message.
    `per_chat_suggestions` triggers a per-chat follow-up suggestion message
    (Add/Decline/Mute buttons) only for chats that have a pending suggestion."""

    async def __call__(
        self,
        text: str,
        *,
        items: list[DigestItem] | None = None,
        per_chat_suggestions: dict[int, SuggestionFooter] | None = None,
    ) -> None: ...


async def digest_loop(settings: Settings, prefs: Preferences, llm: LLM, send: SendText) -> None:
    """Sleeps until the next digest_time, then sends; repeats forever."""
    while True:
        next_run = _next_digest_time(prefs.digest_time)
        sleep_s = max(1.0, (next_run - datetime.now()).total_seconds())
        log.info("next digest at %s (in %.0f min)", next_run.isoformat(), sleep_s / 60)
        await asyncio.sleep(sleep_s)
        try:
            await populate_suggestions(settings, prefs, llm)
            await run_digest(settings, prefs, send)
        except Exception:
            log.exception("digest failed")


async def populate_suggestions(settings: Settings, prefs: Preferences, llm: LLM) -> int:
    """Run both recommendation mechanisms per authorised chat. Returns the count
    of suggestions recorded across all chats this cycle.

    Coexistence rules:
      1. In-domain suggester runs first — higher signal (validated by thumbs).
         If it returns a candidate, record it and skip adjacency for this chat.
      2. Otherwise, fall back to LLM-driven adjacency. Adjacency itself enforces
         a global per-chat cadence cap (~weekly) since the LLM has unlimited
         supply and would otherwise produce a 'want to explore X?' footer
         every digest.
      3. If neither mechanism produces a candidate, the chat gets no suggestion
         this cycle — the digest UX surfaces nothing.

    Per-chat failures don't block other chats — wrap each in try/except so a
    transient LLM hiccup for one chat doesn't break the digest for the rest."""
    recorded = 0
    with connect(settings.db_path) as conn:
        for chat_id in settings.chat_ids:
            try:
                if await _populate_one(conn, chat_id=chat_id, prefs=prefs, llm=llm):
                    recorded += 1
            except Exception:
                log.exception("populating suggestions for chat %s failed", chat_id)
    return recorded


async def _populate_one(
    conn: sqlite3.Connection, *, chat_id: int, prefs: Preferences, llm: LLM
) -> bool:
    """Try in-domain → adjacency for one chat. Returns True if a suggestion
    was recorded."""
    chat_id_str = str(chat_id)
    in_domain = suggest_for_chat(conn, chat_id=chat_id_str, prefs_topics=prefs.topics)
    if in_domain is not None:
        record_for_chat(conn, chat_id=chat_id_str, suggestion=in_domain)
        log.info(
            "recorded in-domain suggestion for chat %s: topic=%r confidence=%.2f",
            chat_id,
            in_domain.topic,
            in_domain.confidence,
        )
        return True
    adjacency = await suggest_for_chat_via_adjacency(
        llm, conn, chat_id=chat_id_str, prefs_topics=prefs.topics
    )
    if adjacency is not None:
        record_for_chat(conn, chat_id=chat_id_str, suggestion=adjacency)
        log.info(
            "recorded adjacency suggestion for chat %s: topic=%r",
            chat_id,
            adjacency.topic,
        )
        return True
    return False


async def run_digest(settings: Settings, prefs: Preferences, send: SendText) -> int:
    """Send the daily digest. Returns the number of articles included (0 if quiet).

    If any chat has a pending taste-discovery suggestion, attach it as a per-chat
    follow-up — sent only to chats with one queued, so chats without one stay
    single-message."""
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
    suggestions = fetch_pending_suggestions(settings)
    await send(msg, items=items, per_chat_suggestions=suggestions)
    with connect(settings.db_path) as conn:
        mark_many_pushed(conn, [r["id"] for r in rows])
    log.info(
        "digest: sent %d articles, %d suggestion follow-ups",
        len(rows),
        len(suggestions),
    )
    return len(rows)


def fetch_pending_suggestions(settings: Settings) -> dict[int, SuggestionFooter]:
    """Snapshot the highest-confidence pending suggestion per authorised chat.
    Empty dict when nobody has one queued — that's the common case, since the
    recommender is gated on engagement signal that takes days to accumulate."""
    out: dict[int, SuggestionFooter] = {}
    with connect(settings.db_path) as conn:
        for chat_id in settings.chat_ids:
            row = pending_suggestion_for(conn, str(chat_id))
            if row is None:
                continue
            try:
                evidence = json.loads(row["evidence_article_ids"] or "[]")
            except json.JSONDecodeError:
                evidence = []
            out[chat_id] = SuggestionFooter(
                suggestion_id=row["id"],
                topic=row["topic"],
                evidence_count=len(evidence) if isinstance(evidence, list) else 0,
            )
    return out


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
