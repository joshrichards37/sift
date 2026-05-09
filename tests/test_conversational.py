from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sift.config import Preferences, SourcePref
from sift.conversational import (
    Intent,
    UndoSnapshot,
    UndoStack,
    _intent_from_json,
    apply_intent,
    persist_prefs,
    serialize_prefs,
)


def _prefs() -> Preferences:
    return Preferences(
        topics="- existing topic",
        exclude_keywords=["crypto"],
        relevance_threshold=7,
        digest_size=10,
        sources=[SourcePref(id="hn", query="x", enabled=True)],
    )


# ── Intent classification (the JSON → Intent half, not the LLM call) ──────


def test_intent_add_topic_normalises_text() -> None:
    intent = _intent_from_json({"kind": "add_topic", "text": "  vLLM releases  "})
    assert intent.kind == "add_topic"
    assert intent.text == "vLLM releases"
    assert "vLLM releases" in intent.summary


def test_intent_add_topic_empty_text_falls_back_to_chat() -> None:
    """Empty text is not actionable — better to fall through to chat than
    silently apply a no-op edit the user has to undo."""
    intent = _intent_from_json({"kind": "add_topic", "text": ""})
    assert intent.kind == "chat"


def test_intent_set_threshold_clamps_out_of_range() -> None:
    """Threshold is 1-10. Out-of-range values from the LLM fall back to chat
    rather than getting clamped silently — surfacing the misclassification
    to the user."""
    assert _intent_from_json({"kind": "set_threshold", "value": 0}).kind == "chat"
    assert _intent_from_json({"kind": "set_threshold", "value": 11}).kind == "chat"
    assert _intent_from_json({"kind": "set_threshold", "value": 6}).kind == "set_threshold"


def test_intent_set_threshold_handles_non_int() -> None:
    assert _intent_from_json({"kind": "set_threshold", "value": "high"}).kind == "chat"


def test_intent_unknown_kind_is_chat() -> None:
    assert _intent_from_json({"kind": "delete_everything"}).kind == "chat"


def test_intent_source_change_passes_through() -> None:
    """Source edits are out of scope for live mutation, but the bot still
    needs to know they were intended so it can show a 'restart required'
    message rather than silently treating it as chat."""
    intent = _intent_from_json({"kind": "source_change"})
    assert intent.kind == "source_change"


# ── apply_intent: in-memory mutation ──────────────────────────────────────


def test_apply_add_topic_appends_bullet() -> None:
    prefs = _prefs()
    apply_intent(prefs, Intent(kind="add_topic", text="new thing"))
    assert "new thing" in prefs.topics
    assert "existing topic" in prefs.topics  # didn't clobber


def test_apply_add_exclude_keyword_dedupes() -> None:
    """Re-adding an existing keyword should be a no-op, not a duplicate."""
    prefs = _prefs()
    apply_intent(prefs, Intent(kind="add_exclude_keyword", text="crypto"))
    assert prefs.exclude_keywords.count("crypto") == 1


def test_apply_set_threshold() -> None:
    prefs = _prefs()
    apply_intent(prefs, Intent(kind="set_threshold", value=8))
    assert prefs.relevance_threshold == 8


def test_apply_set_digest_size() -> None:
    prefs = _prefs()
    apply_intent(prefs, Intent(kind="set_digest_size", value=15))
    assert prefs.digest_size == 15


def test_apply_chat_kind_raises() -> None:
    """apply_intent should never be called with kind=chat — guard it
    explicitly so a routing bug becomes a loud test failure."""
    with pytest.raises(ValueError, match="non-applicable"):
        apply_intent(_prefs(), Intent(kind="chat"))


# ── Atomic write ──────────────────────────────────────────────────────────


def test_persist_prefs_atomic_round_trip(tmp_path: Path) -> None:
    """Write, read back, parse, and confirm the changes survived. Also
    verify the tempfile didn't linger after the rename."""
    prefs = _prefs()
    apply_intent(prefs, Intent(kind="set_threshold", value=8))
    target = tmp_path / "preferences.yaml"
    persist_prefs(prefs, target)

    reloaded = Preferences.model_validate(yaml.safe_load(target.read_text()))
    assert reloaded.relevance_threshold == 8
    # No leftover .tmp files in the dir
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


# ── Undo ──────────────────────────────────────────────────────────────────


def test_undo_stack_lifo_order() -> None:
    stack = UndoStack(cap=3)
    stack.push(UndoSnapshot(yaml_text="a", summary="first"))
    stack.push(UndoSnapshot(yaml_text="b", summary="second"))
    stack.push(UndoSnapshot(yaml_text="c", summary="third"))
    assert stack.pop().summary == "third"
    assert stack.pop().summary == "second"
    assert stack.pop().summary == "first"
    assert stack.pop() is None


def test_undo_stack_caps_oldest_evicted() -> None:
    stack = UndoStack(cap=2)
    stack.push(UndoSnapshot(yaml_text="a", summary="first"))
    stack.push(UndoSnapshot(yaml_text="b", summary="second"))
    stack.push(UndoSnapshot(yaml_text="c", summary="third"))
    assert len(stack) == 2
    assert stack.pop().summary == "third"
    assert stack.pop().summary == "second"


def test_serialize_round_trips() -> None:
    """Round-trip a Preferences through serialize → safe_load → validate.
    Regression guard: if model_dump output stops being valid input,
    persist_prefs would silently corrupt the user's file."""
    prefs = _prefs()
    text = serialize_prefs(prefs)
    reloaded = Preferences.model_validate(yaml.safe_load(text))
    assert reloaded.topics == prefs.topics
    assert reloaded.exclude_keywords == prefs.exclude_keywords
