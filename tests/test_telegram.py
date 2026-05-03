from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sift.config import Settings
from sift.storage import connect, init_db
from sift.telegram_bot import (
    EXPAND_CALLBACK,
    Bot,
    _build_collapsed_keyboard,
    _build_suggestion_keyboard,
    _build_thumbs_keyboard,
    _chunk,
    _parse_feedback_callback,
    _parse_suggestion_callback,
)

LIMIT = 4096  # Telegram message size limit


def _settings(monkeypatch: pytest.MonkeyPatch, *, db_path: Path) -> Settings:
    """Minimal Settings for handler tests — no on-disk .env, no LLM env override."""
    for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "OLLAMA_URL", "OLLAMA_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test:token")
    monkeypatch.setenv("OWNER_CHAT_ID", "42")
    monkeypatch.setenv("AUTHORIZED_CHAT_IDS", "")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.db_path = db_path
    return s


def _make_bot(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> Bot:
    """Construct a Bot for handler tests. The Application is built but never
    started — handlers are invokable directly without networking."""
    from sift.config import Preferences, SourcePref
    from sift.llm import LLM

    settings = _settings(monkeypatch, db_path=db_path)
    prefs = Preferences(topics="t", sources=[SourcePref(id="hn", query="x")])
    llm = LLM(base_url="http://test/v1", api_key="x", model="m")
    return Bot(settings=settings, prefs=prefs, llm=llm)


def test_chunk_short_message_returned_unchanged() -> None:
    text = "small message"
    assert _chunk(text, LIMIT) == [text]


def test_chunk_splits_on_paragraph_boundary() -> None:
    """The whole point of _chunk: split on \\n\\n so HTML inside one digest
    entry stays in a single chunk. If we ever lose the paragraph-aware split,
    multi-chunk messages can break HTML formatting mid-tag."""
    para = "x" * 2000
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = _chunk(text, LIMIT)

    # Limit respected
    assert all(len(c) <= LIMIT for c in chunks)
    # Got more than one chunk (input is bigger than the limit)
    assert len(chunks) > 1
    # Critical invariant: rejoining chunks with the paragraph delimiter
    # recovers the original. If the splitter ever cuts mid-paragraph, this
    # assertion breaks.
    assert "\n\n".join(chunks) == text


def test_chunk_handles_oversized_paragraph_with_hard_cut() -> None:
    """A single paragraph longer than the limit forces a hard cut — we'd rather
    deliver the message split mid-paragraph than not at all."""
    text = "y" * (LIMIT + 500)  # one long paragraph, no breaks
    chunks = _chunk(text, LIMIT)
    assert len(chunks) >= 2
    assert all(len(c) <= LIMIT for c in chunks)
    # All content preserved (the function rstrips/lstrips newlines, so for a
    # newline-free input nothing is lost)
    assert "".join(chunks) == text


def test_chunk_prefers_paragraph_then_line_then_hard_cut() -> None:
    """Three-tier fallback: paragraph break > line break > hard limit cut."""
    block = "z" * 3000
    # No paragraph breaks, but plenty of single newlines
    text = "\n".join([block, block, block])
    chunks = _chunk(text, LIMIT)
    assert all(len(c) <= LIMIT for c in chunks)
    # Should split on the single newlines, so we end up with multiple chunks
    assert len(chunks) >= 2


# --- Thumbs keyboard ---


def test_build_collapsed_keyboard_is_single_expand_button() -> None:
    """Default state under a digest: one button. Tapping fires fb:expand and the
    bot swaps in the per-item keyboard. Keeps the resting state clean."""
    kb = _build_collapsed_keyboard()
    assert len(kb.inline_keyboard) == 1
    assert len(kb.inline_keyboard[0]) == 1
    assert kb.inline_keyboard[0][0].callback_data == EXPAND_CALLBACK


def test_build_thumbs_keyboard_one_row_per_item() -> None:
    """Expanded state: n items → n rows, [👍 N][👎 N] per row. Pairing each
    button unambiguously to its number — clearer than packing items 2-per-row."""
    items = [(1, "aaa"), (2, "bbb"), (3, "ccc")]
    kb = _build_thumbs_keyboard(items)
    assert len(kb.inline_keyboard) == 3
    for row in kb.inline_keyboard:
        assert len(row) == 2
    assert kb.inline_keyboard[0][0].text == "👍 1"
    assert kb.inline_keyboard[0][1].text == "👎 1"
    assert kb.inline_keyboard[2][0].text == "👍 3"
    assert kb.inline_keyboard[2][1].text == "👎 3"


def test_build_thumbs_keyboard_callback_data_format() -> None:
    """callback_data must encode (article_id, rating) so the handler can decode
    without external lookup. Format is fb:<article_id>:<+1|-1> — round-trips
    cleanly through _parse_feedback_callback."""
    kb = _build_thumbs_keyboard([(1, "abc123")])
    assert kb.inline_keyboard[0][0].callback_data == "fb:abc123:+1"
    assert kb.inline_keyboard[0][1].callback_data == "fb:abc123:-1"
    # And the parser is the inverse.
    assert _parse_feedback_callback("fb:abc123:+1") == ("abc123", 1)
    assert _parse_feedback_callback("fb:abc123:-1") == ("abc123", -1)


def test_build_thumbs_keyboard_empty_items_returns_empty_keyboard() -> None:
    """Defensive: edge case if expand fires with no items recorded.
    Still valid markup."""
    kb = _build_thumbs_keyboard([])
    assert kb.inline_keyboard == ()


# --- Callback parser ---


def test_parse_feedback_callback_rejects_wrong_prefix() -> None:
    """Anything not prefixed 'fb:' is not for us — returning None lets the bot
    answer with a graceful 'bad data' rather than recording bogus feedback."""
    assert _parse_feedback_callback("xx:abc:+1") is None
    assert _parse_feedback_callback("abc") is None


def test_parse_feedback_callback_rejects_wrong_arity() -> None:
    """Too few or too many colons → None. Don't try to recover."""
    assert _parse_feedback_callback("fb:abc") is None
    assert _parse_feedback_callback("fb:abc:+1:extra") is None


def test_parse_feedback_callback_rejects_invalid_rating() -> None:
    """Only +1 and -1 are valid. Anything else (including '0', '2', 'up') → None."""
    assert _parse_feedback_callback("fb:abc:0") is None
    assert _parse_feedback_callback("fb:abc:2") is None
    assert _parse_feedback_callback("fb:abc:up") is None


def test_parse_feedback_callback_rejects_empty_article_id() -> None:
    """Defensive: empty article_id segment shouldn't pass through to record_feedback."""
    assert _parse_feedback_callback("fb::+1") is None


# --- Callback handler ---


async def test_on_feedback_records_thumbs_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Happy path: an authorised user clicks 👍 on item N, the corresponding
    article_id ends up in the feedback table with rating=+1. The article must
    exist first — feedback.article_id is an FK with ON DELETE CASCADE — which
    is fine in production because callbacks only ever fire on articles we sent."""
    from sift.storage import insert_article

    db = tmp_path / "test.db"
    init_db(db)
    with connect(db) as conn:
        aid = insert_article(conn, source_id="hn", url="https://x/1", title="t")
    assert aid is not None
    bot = _make_bot(monkeypatch, db)
    answer = AsyncMock()
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            from_user=SimpleNamespace(id=42),  # owner from _settings
            data=f"fb:{aid}:+1",
            answer=answer,
        )
    )
    await bot._on_feedback(update, None)
    with connect(db) as conn:
        rows = conn.execute(
            "SELECT article_id, rating FROM feedback WHERE article_id = ?", (aid,)
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["rating"] == 1
    answer.assert_awaited_once()


async def test_on_feedback_rejects_unauthorised_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Security boundary: callback queries don't go through the message-level
    chat allowlist, so the handler MUST re-check user_id. A spoofed callback
    from a non-allowlisted user must not write to feedback."""
    db = tmp_path / "test.db"
    init_db(db)
    bot = _make_bot(monkeypatch, db)
    answer = AsyncMock()
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            from_user=SimpleNamespace(id=999),  # not on allowlist
            data="fb:abc123:+1",
            answer=answer,
        )
    )
    await bot._on_feedback(update, None)
    with connect(db) as conn:
        rows = conn.execute("SELECT * FROM feedback").fetchall()
    assert rows == []
    answer.assert_awaited_once()  # always answer, even on rejection (15s timeout)


async def test_on_feedback_handles_malformed_callback_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An authorised user with bad callback data — don't write feedback, but
    still answer the query so the spinner clears."""
    db = tmp_path / "test.db"
    init_db(db)
    bot = _make_bot(monkeypatch, db)
    answer = AsyncMock()
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            from_user=SimpleNamespace(id=42),
            data="garbage",
            answer=answer,
        )
    )
    await bot._on_feedback(update, None)
    with connect(db) as conn:
        rows = conn.execute("SELECT * FROM feedback").fetchall()
    assert rows == []
    answer.assert_awaited_once()


# --- Expand callback ---


async def test_expand_callback_swaps_in_full_keyboard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tapping 'Rate items' edits the message reply markup to the full per-item
    keyboard. The bot recovers items from in-memory state keyed by message_id."""
    db = tmp_path / "test.db"
    init_db(db)
    bot = _make_bot(monkeypatch, db)
    # Pretend we sent a digest with 3 items earlier. message_id=999.
    bot._digest_items[999] = [(1, "aid1"), (2, "aid2"), (3, "aid3")]
    edit_markup = AsyncMock()
    answer = AsyncMock()
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            from_user=SimpleNamespace(id=42),
            data=EXPAND_CALLBACK,
            message=SimpleNamespace(message_id=999),
            edit_message_reply_markup=edit_markup,
            answer=answer,
        )
    )
    await bot._on_feedback(update, None)
    edit_markup.assert_awaited_once()
    # The new keyboard should have one row per item (3 rows, 2 buttons each).
    new_markup = edit_markup.await_args.kwargs["reply_markup"]
    assert len(new_markup.inline_keyboard) == 3
    assert new_markup.inline_keyboard[0][0].callback_data == "fb:aid1:+1"
    answer.assert_awaited_once()


async def test_expand_callback_says_expired_when_state_lost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the bot restarted (or this digest's entry got pruned by the memory
    cap), tapping Rate must answer politely rather than silently failing or
    crashing. No markup edit attempted."""
    db = tmp_path / "test.db"
    init_db(db)
    bot = _make_bot(monkeypatch, db)
    # No entry in _digest_items for message_id=12345 — simulates restart.
    edit_markup = AsyncMock()
    answer = AsyncMock()
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            from_user=SimpleNamespace(id=42),
            data=EXPAND_CALLBACK,
            message=SimpleNamespace(message_id=12345),
            edit_message_reply_markup=edit_markup,
            answer=answer,
        )
    )
    await bot._on_feedback(update, None)
    edit_markup.assert_not_awaited()
    answer.assert_awaited_once()
    # The user-facing answer should mention the digest expired/restart situation.
    msg = answer.await_args.args[0] if answer.await_args.args else ""
    assert "no longer rateable" in msg.lower() or "restart" in msg.lower()


def test_prune_digest_memory_drops_oldest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When _digest_items exceeds the cap, the OLDEST message_ids get evicted.
    Stops the dict from growing unboundedly across many days of digests."""
    from sift.telegram_bot import DIGEST_MEMORY_CAP

    db = tmp_path / "test.db"
    init_db(db)
    bot = _make_bot(monkeypatch, db)
    # Fill past the cap with monotonic message_ids.
    for mid in range(DIGEST_MEMORY_CAP + 5):
        bot._digest_items[mid] = [(1, "x")]
    bot._prune_digest_memory()
    assert len(bot._digest_items) == DIGEST_MEMORY_CAP
    # The oldest 5 (lowest message_ids) should be gone.
    assert min(bot._digest_items) == 5


# --- Suggestion keyboard + parser ---


def test_build_suggestion_keyboard_three_buttons() -> None:
    """One row, three buttons: Add / Decline / Mute. callback_data encodes the
    suggestion id so the handler can resolve it back to a row."""
    kb = _build_suggestion_keyboard(suggestion_id=42)
    assert len(kb.inline_keyboard) == 1
    assert len(kb.inline_keyboard[0]) == 3
    assert kb.inline_keyboard[0][0].callback_data == "sg:42:add"
    assert kb.inline_keyboard[0][1].callback_data == "sg:42:decline"
    assert kb.inline_keyboard[0][2].callback_data == "sg:42:mute"


def test_parse_suggestion_callback_round_trips() -> None:
    assert _parse_suggestion_callback("sg:7:add") == (7, "add")
    assert _parse_suggestion_callback("sg:99:decline") == (99, "decline")
    assert _parse_suggestion_callback("sg:1:mute") == (1, "mute")


def test_parse_suggestion_callback_rejects_malformed() -> None:
    """Non-int id, unknown action, wrong prefix, wrong arity all → None."""
    assert _parse_suggestion_callback("sg:abc:add") is None
    assert _parse_suggestion_callback("sg:1:upvote") is None
    assert _parse_suggestion_callback("xx:1:add") is None
    assert _parse_suggestion_callback("sg:1") is None
    assert _parse_suggestion_callback("sg:1:add:extra") is None


# --- Suggestion callback handler ---


async def _seed_suggestion(db: Path, *, chat_id: str, topic: str) -> int:
    """Helper: insert a pending suggestion_candidates row and return its id."""
    from sift.storage import record_suggestion_candidate

    with connect(db) as conn:
        return record_suggestion_candidate(
            conn,
            chat_id=chat_id,
            topic=topic,
            confidence=0.8,
            evidence_article_ids=["a1", "a2", "a3"],
        )


async def test_on_suggestion_add_records_added_and_updates_in_memory_prefs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Add appends to in-memory prefs.topics so the LLM uses the new topic in
    the very next score call. Persistence to preferences.yaml is left to the
    user (acknowledgment shows a copy-pasteable snippet) to avoid clobbering
    hand-written comments in their file."""
    from sift.storage import last_response_for_topic

    db = tmp_path / "test.db"
    init_db(db)
    sid = await _seed_suggestion(db, chat_id="42", topic="post-training")
    bot = _make_bot(monkeypatch, db)
    original_topics = bot.prefs.topics
    edit = AsyncMock()
    answer = AsyncMock()
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            from_user=SimpleNamespace(id=42),
            data=f"sg:{sid}:add",
            edit_message_text=edit,
            answer=answer,
        )
    )
    await bot._on_suggestion(update, None)
    # Storage state: response recorded as 'added'.
    with connect(db) as conn:
        last = last_response_for_topic(conn, chat_id="42", topic="post-training")
    assert last is not None
    assert last["response"] == "added"
    # In-memory prefs include the new topic now.
    assert "post-training" in bot.prefs.topics
    assert bot.prefs.topics != original_topics
    # User got an acknowledgment with the copy-pasteable snippet.
    edit.assert_awaited_once()
    ack_text = edit.await_args.kwargs["text"]
    assert "Added" in ack_text and "post-training" in ack_text
    assert "preferences.yaml" in ack_text
    answer.assert_awaited_once()


async def test_on_suggestion_decline_records_declined(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Decline records 'declined' (recommender enforces 30-day cooldown).
    In-memory prefs are unchanged."""
    from sift.storage import last_response_for_topic

    db = tmp_path / "test.db"
    init_db(db)
    sid = await _seed_suggestion(db, chat_id="42", topic="crypto")
    bot = _make_bot(monkeypatch, db)
    before = bot.prefs.topics
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            from_user=SimpleNamespace(id=42),
            data=f"sg:{sid}:decline",
            edit_message_text=AsyncMock(),
            answer=AsyncMock(),
        )
    )
    await bot._on_suggestion(update, None)
    with connect(db) as conn:
        last = last_response_for_topic(conn, chat_id="42", topic="crypto")
    assert last is not None
    assert last["response"] == "declined"
    assert bot.prefs.topics == before  # decline must not pollute prefs


async def test_on_suggestion_mute_records_muted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mute records 'muted' (recommender treats this as permanent — 'Mute
    permanently must actually mean it' was the explicit design constraint)."""
    from sift.storage import last_response_for_topic

    db = tmp_path / "test.db"
    init_db(db)
    sid = await _seed_suggestion(db, chat_id="42", topic="nfts")
    bot = _make_bot(monkeypatch, db)
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            from_user=SimpleNamespace(id=42),
            data=f"sg:{sid}:mute",
            edit_message_text=AsyncMock(),
            answer=AsyncMock(),
        )
    )
    await bot._on_suggestion(update, None)
    with connect(db) as conn:
        last = last_response_for_topic(conn, chat_id="42", topic="nfts")
    assert last is not None
    assert last["response"] == "muted"


async def test_on_suggestion_rejects_unauthorised_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same security boundary as the thumbs handler: callback_query bypasses
    the message-level chat allowlist filter, so the handler MUST re-check user_id."""
    db = tmp_path / "test.db"
    init_db(db)
    sid = await _seed_suggestion(db, chat_id="42", topic="x")
    bot = _make_bot(monkeypatch, db)
    edit = AsyncMock()
    answer = AsyncMock()
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            from_user=SimpleNamespace(id=999),  # not on allowlist
            data=f"sg:{sid}:add",
            edit_message_text=edit,
            answer=answer,
        )
    )
    await bot._on_suggestion(update, None)
    edit.assert_not_awaited()
    answer.assert_awaited_once()
    # No response recorded — db state unchanged
    with connect(db) as conn:
        row = conn.execute(
            "SELECT response FROM suggestion_candidates WHERE id = ?", (sid,)
        ).fetchone()
    assert row["response"] is None
