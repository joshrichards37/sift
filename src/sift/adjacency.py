"""LLM-driven topic adjacency suggester.

Mechanism #2 of the recommendation engine. Where the in-domain suggester
(sift.recommender) discovers topics from behaviour signal — articles the user
thumbed up that aren't yet in their stated interests — adjacency reasons over
the user's stated interests directly and proposes topics they *might* also
want to follow.

Two key differences from in-domain:
  - Works on day one, no warmup. No thumbs needed.
  - Speculative — adjacency is a guess, not validated engagement.

The supply curves explain why each is gated differently:
  - In-domain has natural scarcity (needs 3+ thumbs in a topic), so
    one-per-digest with per-topic cooldown is enough.
  - Adjacency has unlimited supply (an LLM can always reason out another
    adjacent topic). Without a global per-chat cadence cap, the digest would
    sprout a 'want to explore X?' footer every single day. The cap is a
    nag-ware guardrail, not a quality filter.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta

from sift.llm import LLM
from sift.recommender import Suggestion, topic_blocked_by_prior_response

log = logging.getLogger(__name__)


ADJACENCY_SYSTEM = """You propose adjacent topics for a user's news feed.

Given a list of their stated interests, suggest 3-5 topics they might also
want to follow that are NOT already in their list. Topics should be:
- Specific enough to be useful as feed labels (e.g. "model interpretability"
  not "AI"; "rust async runtime" not "Rust")
- Adjacent — connected by craft, community, or theme; not random tangents
- Lowercase, 1-3 words each

Reply with strict JSON only: {"topics": ["topic1", "topic2", ...]}
No markdown, no preamble."""


# Default global cadence cap: at most one suggestion of any kind per chat per
# week. Tunable via cooldown_days kwarg for testing / future calibration.
DEFAULT_GLOBAL_COOLDOWN_DAYS = 7

# Decline cooldown reuses the in-domain default (30 days). A topic the user
# explicitly said no to shouldn't bounce back via adjacency next week either.
DEFAULT_DECLINE_COOLDOWN_DAYS = 30

# Adjacency confidence is a fixed sub-1.0 value. The in-domain suggester scores
# 1.0 for 10 samples × 100% positive, so this sits below it — when the digest
# UX surfaces "highest pending suggestion" via pending_suggestion_for, in-domain
# wins the race. Mostly cosmetic since the two paths are gated to not coexist
# in the same digest cycle anyway.
ADJACENCY_CONFIDENCE = 0.5


async def suggest_adjacent(
    llm: LLM, prefs_topics: str, exclude_topics: list[str] | None = None
) -> list[str]:
    """Ask the LLM for adjacent topics. Returns a cleaned list — caller still
    must filter against in-prefs / mute / decline state, since the model isn't
    trusted to honour the exclude_topics hint perfectly."""
    exclude_blob = ""
    if exclude_topics:
        exclude_blob = (
            "\n\nDo NOT propose any of these (already in interests or rejected): "
            + ", ".join(exclude_topics)
        )
    user_msg = f"User interests:\n{prefs_topics}{exclude_blob}"
    raw = await llm.chat(ADJACENCY_SYSTEM, [], user_msg)
    return _parse_topics(raw)


def cadence_blocks_chat(
    conn: sqlite3.Connection, *, chat_id: str, now: datetime, cooldown_days: int
) -> bool:
    """True if any suggestion (in-domain OR adjacency) was created within the
    cooldown window for this chat. Both types count toward the cap so the user
    gets at most one suggestion per cooldown period regardless of mechanism."""
    cutoff = (now - timedelta(days=cooldown_days)).isoformat()
    row = conn.execute(
        """
        SELECT 1 FROM suggestion_candidates
        WHERE chat_id = ? AND created_at >= ?
        LIMIT 1
        """,
        (chat_id, cutoff),
    ).fetchone()
    return row is not None


async def suggest_for_chat_via_adjacency(
    llm: LLM,
    conn: sqlite3.Connection,
    *,
    chat_id: str,
    prefs_topics: str,
    cooldown_days: int = DEFAULT_GLOBAL_COOLDOWN_DAYS,
    decline_cooldown_days: int = DEFAULT_DECLINE_COOLDOWN_DAYS,
    now: datetime | None = None,
) -> Suggestion | None:
    """Returns a Suggestion if adjacency is appropriate now, else None.

    Order of checks:
      1. Cadence cap — any suggestion within cooldown window blocks (cheapest;
         skip the LLM call entirely if recent activity already used the slot).
      2. LLM call — propose 3-5 adjacent topics.
      3. Filter each candidate against in-prefs (case-insensitive substring),
         muted (permanent), declined (decline cooldown). Return first that
         passes."""
    now = now or datetime.now(UTC)
    if cadence_blocks_chat(conn, chat_id=chat_id, now=now, cooldown_days=cooldown_days):
        return None
    try:
        candidates = await suggest_adjacent(llm, prefs_topics)
    except Exception:
        log.exception("adjacency LLM call failed for chat %s", chat_id)
        return None
    prefs_lower = (prefs_topics or "").lower()
    for topic in candidates:
        if topic in prefs_lower:
            continue
        if topic_blocked_by_prior_response(
            conn,
            chat_id=chat_id,
            topic=topic,
            now=now,
            decline_cooldown_days=decline_cooldown_days,
        ):
            continue
        return Suggestion(
            topic=topic,
            confidence=ADJACENCY_CONFIDENCE,
            evidence_article_ids=[],  # adjacency has no behavioural evidence
        )
    return None


def _parse_topics(raw: str) -> list[str]:
    """Decode the LLM's JSON response. Tolerant of typical model failure modes:
    missing 'topics' key, non-list value, non-string elements, leading whitespace."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("adjacency returned non-JSON: %r", raw[:200])
        return []
    topics = parsed.get("topics") if isinstance(parsed, dict) else None
    if not isinstance(topics, list):
        return []
    out: list[str] = []
    for t in topics:
        if not isinstance(t, str):
            continue
        cleaned = t.strip().lower()[:60]
        if cleaned:
            out.append(cleaned)
    return out
