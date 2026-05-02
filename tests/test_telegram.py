from __future__ import annotations

from sift.telegram_bot import _chunk

LIMIT = 4096  # Telegram message size limit


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
