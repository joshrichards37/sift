"""Taste-discovery suggester.

Reads the user's recent thumbs feedback joined with the article topic tags written
during scoring, and identifies topics the user reliably engages with that aren't
in their stated preferences. The result is recorded in suggestion_candidates so the
digest UX can surface it to the user with Add / Decline / Mute buttons.

This is mechanism #1 of the recommendation engine — purely local, no cross-user
collaborative signal. Works from a single user's own thumbs.

The actual digest-side surfacing is intentionally not in this module — we only
*produce* suggestions here. The Telegram digest layer reads pending_suggestion_for()
when assembling the next message.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sift.storage import (
    article_topic_tags,
    last_response_for_topic,
    record_suggestion_candidate,
)

log = logging.getLogger(__name__)


# The hard floors (min_samples + min_positive_ratio) gate eligibility; the
# confidence score is then used for ranking when multiple candidates exist.
# min_confidence defaults to 0 — start permissive, tune up later if real
# behaviour is noisy. Easier to tighten than to discover users never see suggestions.
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_MIN_SAMPLES = 3
DEFAULT_MIN_POSITIVE_RATIO = 0.7
DEFAULT_DECLINE_COOLDOWN_DAYS = 30
DEFAULT_MIN_CONFIDENCE = 0.0


@dataclass(slots=True, frozen=True)
class Suggestion:
    topic: str
    confidence: float
    evidence_article_ids: list[str]


def suggest_for_chat(
    conn: sqlite3.Connection,
    *,
    chat_id: str,
    prefs_topics: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_positive_ratio: float = DEFAULT_MIN_POSITIVE_RATIO,
    decline_cooldown_days: int = DEFAULT_DECLINE_COOLDOWN_DAYS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    now: datetime | None = None,
) -> Suggestion | None:
    """Compute the top-1 topic suggestion for this chat, or None.

    Pure function over the DB — does not write. Pass the result to
    record_for_chat() if you want to persist it.
    """
    now = now or datetime.now(UTC)
    cutoff = (now - timedelta(days=lookback_days)).isoformat()

    rows = conn.execute(
        """
        SELECT a.id AS article_id, a.topic_tags, f.rating
        FROM feedback f
        JOIN articles a ON a.id = f.article_id
        WHERE f.created_at >= ?
          AND a.topic_tags IS NOT NULL
        """,
        (cutoff,),
    ).fetchall()

    # tag -> list of (article_id, rating)
    by_tag: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in rows:
        for tag in article_topic_tags(r["topic_tags"]):
            by_tag[tag].append((r["article_id"], int(r["rating"])))

    prefs_lower = (prefs_topics or "").lower()

    candidates: list[Suggestion] = []
    for tag, samples in by_tag.items():
        if tag in prefs_lower:
            continue  # already a stated interest — no point suggesting it
        positive = [aid for aid, rating in samples if rating > 0]
        n = len(samples)
        if n < min_samples:
            continue
        positive_ratio = len(positive) / n
        if positive_ratio < min_positive_ratio:
            continue
        if _topic_blocked_by_prior_response(
            conn,
            chat_id=chat_id,
            topic=tag,
            now=now,
            decline_cooldown_days=decline_cooldown_days,
        ):
            continue
        confidence = positive_ratio * min(1.0, n / 10.0)
        if confidence < min_confidence:
            continue
        # Cap evidence to keep the row small; positives are what justify the suggestion.
        candidates.append(
            Suggestion(topic=tag, confidence=confidence, evidence_article_ids=positive[:10])
        )

    if not candidates:
        return None
    candidates.sort(key=lambda s: s.confidence, reverse=True)
    return candidates[0]


def record_for_chat(conn: sqlite3.Connection, *, chat_id: str, suggestion: Suggestion) -> int:
    """Persist a suggestion. Separated from suggest_for_chat so callers can dry-run."""
    return record_suggestion_candidate(
        conn,
        chat_id=chat_id,
        topic=suggestion.topic,
        confidence=suggestion.confidence,
        evidence_article_ids=suggestion.evidence_article_ids,
    )


def _topic_blocked_by_prior_response(
    conn: sqlite3.Connection,
    *,
    chat_id: str,
    topic: str,
    now: datetime,
    decline_cooldown_days: int,
) -> bool:
    """'muted' is permanent. 'declined' has a cooldown. 'added' shouldn't happen
    — if the topic was added, the user's prefs would now contain it and the
    earlier in-prefs check would already have skipped this tag — but treat it as
    blocked anyway for safety."""
    last = last_response_for_topic(conn, chat_id=chat_id, topic=topic)
    if last is None:
        return False
    response = last["response"]
    if response in ("muted", "added"):
        return True
    if response == "declined":
        responded_at = datetime.fromisoformat(last["responded_at"])
        return (now - responded_at) < timedelta(days=decline_cooldown_days)
    return False
