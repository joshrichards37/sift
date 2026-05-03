from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sift.config import Settings
from sift.storage import connect, init_db
from sift.telegram_bot import (
    Bot,
    _build_thumbs_keyboard,
    _chunk,
    _parse_feedback_callback,
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


def test_build_thumbs_keyboard_four_buttons_per_row() -> None:
    """Compact layout: two items packed per row, each as [👍 N][👎 N]. Halves
    keyboard height vs one-item-per-row. Last row may be 2 buttons if items is odd."""
    items = [(1, "aaa"), (2, "bbb"), (3, "ccc")]
    kb = _build_thumbs_keyboard(items)
    assert len(kb.inline_keyboard) == 2  # ceil(3 / 2)
    # First row packs items 1 and 2 → 4 buttons
    assert len(kb.inline_keyboard[0]) == 4
    assert kb.inline_keyboard[0][0].text == "👍 1"
    assert kb.inline_keyboard[0][1].text == "👎 1"
    assert kb.inline_keyboard[0][2].text == "👍 2"
    assert kb.inline_keyboard[0][3].text == "👎 2"
    # Second row only has item 3 → 2 buttons
    assert len(kb.inline_keyboard[1]) == 2
    assert kb.inline_keyboard[1][0].text == "👍 3"
    assert kb.inline_keyboard[1][1].text == "👎 3"


def test_build_thumbs_keyboard_full_ten_items_packs_to_five_rows() -> None:
    """The default digest_size=10 case — 10 items pack into exactly 5 rows of 4."""
    items = [(i, f"id{i}") for i in range(1, 11)]
    kb = _build_thumbs_keyboard(items)
    assert len(kb.inline_keyboard) == 5
    assert all(len(row) == 4 for row in kb.inline_keyboard)


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
    """Defensive: edge case if a 'digest' fires with no items. Still valid markup."""
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
