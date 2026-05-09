"""Conversational config editing for the Telegram bot.

Free-text messages from the user can either be questions about recent
articles (handled in telegram_bot._on_text via LLM.chat) or *config
edits* — "follow vLLM releases", "lower the threshold to 6", "mute
funding announcements". This module handles classification and the
mutation primitives; the bot owns the confirmation UX.

Scope for v1: only fields that take effect immediately without a
scheduler restart — `topics`, `exclude_keywords`, `relevance_threshold`,
`digest_size`. Source add/remove requires rebuilding the source list and
restarting the scheduler, so we surface a "restart required" message
for those classifications and let the user edit `preferences.yaml`
manually. Filed as a follow-up.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from sift.config import Preferences
from sift.llm import LLM

log = logging.getLogger(__name__)

# Cap on the undo ring. 10 is enough for a chat session of edits without
# letting memory drift unboundedly across days of uptime.
UNDO_CAP = 10


IntentKind = Literal[
    "add_topic",
    "add_exclude_keyword",
    "set_threshold",
    "set_digest_size",
    "source_change",
    "chat",
    "unknown",
]


@dataclass
class Intent:
    kind: IntentKind
    # One of these is populated depending on `kind`. Kept as separate optional
    # fields rather than a union so the bot can pattern-match without isinstance.
    text: str | None = None  # for add_topic / add_exclude_keyword
    value: int | None = None  # for set_threshold / set_digest_size
    summary: str = ""  # human-readable diff line for the confirmation prompt


CLASSIFY_SYSTEM = """You classify a user's chat message to a personal news bot. \
The message is either a question (kind=chat) or a config edit. Output strict JSON only:

  { "kind": "add_topic", "text": "<one short topic bullet>" }
    — user wants to track something new. Pick a concrete bullet, name people/products.
    "follow vLLM releases" → { "kind": "add_topic", "text": "vLLM releases and changelogs" }

  { "kind": "add_exclude_keyword", "text": "<keyword>" }
    — user wants to silence articles containing a word. Single word or short phrase.
    "mute funding announcements" → { "kind": "add_exclude_keyword", "text": "funding" }

  { "kind": "set_threshold", "value": <int 1-10> }
    — user wants to change the relevance threshold (cutoff for the digest).
    "lower the threshold to 6" → { "kind": "set_threshold", "value": 6 }

  { "kind": "set_digest_size", "value": <int 1-50> }
    — user wants more or fewer items in the daily digest.
    "make digests 15 items" → { "kind": "set_digest_size", "value": 15 }

  { "kind": "source_change" }
    — user wants to add, remove, or modify a source (RSS feed, GitHub repo, subreddit, etc).
    These require a restart in the current version, so we surface a polite refusal.
    "add the vllm github repo" → { "kind": "source_change" }
    "stop polling reddit" → { "kind": "source_change" }

  { "kind": "chat" }
    — anything else: questions about recent articles, opinions, conversation.
    "what's the latest on Claude Code?" → { "kind": "chat" }

When ambiguous between an edit and chat, prefer chat. False positives on edits are
worse than false negatives — the user can rephrase if needed."""


async def classify_intent(llm: LLM, message: str) -> Intent:
    """Ask the LLM to classify a free-text message. Returns an Intent that
    the bot can use to render a confirmation keyboard or fall through to
    chat. Failure modes (parse error, unrecognized kind, missing fields)
    fall back to kind='chat' so a misclassification never blocks the user
    from asking a question."""
    try:
        resp = await llm.client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
    except Exception:
        log.exception("intent classification failed; defaulting to chat")
        return Intent(kind="chat")
    return _intent_from_json(parsed)


def _intent_from_json(parsed: dict) -> Intent:
    kind_raw = parsed.get("kind", "chat")
    if kind_raw == "add_topic":
        text = (parsed.get("text") or "").strip()
        if not text:
            return Intent(kind="chat")
        return Intent(kind="add_topic", text=text, summary=f"add topic: “{text}”")
    if kind_raw == "add_exclude_keyword":
        text = (parsed.get("text") or "").strip()
        if not text:
            return Intent(kind="chat")
        return Intent(
            kind="add_exclude_keyword",
            text=text,
            summary=f"exclude articles containing: “{text}”",
        )
    if kind_raw == "set_threshold":
        try:
            v = int(parsed.get("value", 0))
        except (TypeError, ValueError):
            return Intent(kind="chat")
        if not 1 <= v <= 10:
            return Intent(kind="chat")
        return Intent(kind="set_threshold", value=v, summary=f"set relevance threshold to {v}/10")
    if kind_raw == "set_digest_size":
        try:
            v = int(parsed.get("value", 0))
        except (TypeError, ValueError):
            return Intent(kind="chat")
        if not 1 <= v <= 50:
            return Intent(kind="chat")
        return Intent(kind="set_digest_size", value=v, summary=f"set digest size to {v} items")
    if kind_raw == "source_change":
        return Intent(kind="source_change", summary="source add/remove (restart required)")
    return Intent(kind="chat")


# ── Mutation ──────────────────────────────────────────────────────────────


def apply_intent(prefs: Preferences, intent: Intent) -> None:
    """Mutate the in-memory Preferences object. The Telegram bot holds the
    same reference, so the next LLM scoring call picks up the change without
    a restart. Caller is responsible for persisting via persist_prefs()."""
    if intent.kind == "add_topic":
        assert intent.text
        prefs.topics = f"{prefs.topics.rstrip()}\n  - {intent.text}"
    elif intent.kind == "add_exclude_keyword":
        assert intent.text
        if intent.text not in prefs.exclude_keywords:
            prefs.exclude_keywords.append(intent.text)
    elif intent.kind == "set_threshold":
        assert intent.value is not None
        prefs.relevance_threshold = intent.value
    elif intent.kind == "set_digest_size":
        assert intent.value is not None
        prefs.digest_size = intent.value
    else:
        raise ValueError(f"apply_intent called with non-applicable kind: {intent.kind}")


def serialize_prefs(prefs: Preferences) -> str:
    """Render Preferences as YAML. Note: hand-written comments and field
    ordering in the original preferences.yaml are NOT preserved — this is a
    deliberate trade for simpler code. Users who care about formatting
    should keep editing the file manually instead of via chat."""
    data = prefs.model_dump(exclude_none=True)
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=120)


def persist_prefs(prefs: Preferences, path: Path) -> None:
    """Atomic write: dump to a sibling tempfile, fsync, then rename. A crash
    or kill mid-write leaves the original file intact rather than truncated."""
    payload = serialize_prefs(prefs)
    tmp_dir = path.parent
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=tmp_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Clean up the tempfile if rename never happened.
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


# ── Undo ──────────────────────────────────────────────────────────────────


@dataclass
class UndoSnapshot:
    """One pre-edit state snapshot for /undo. We save the full Preferences
    YAML rather than a diff so undo is a simple deserialize-and-replace."""

    yaml_text: str
    summary: str  # what the next-applied edit was, for /undo's reply


class UndoStack:
    """Bounded LIFO of pre-edit snapshots. Trims the oldest when full so
    long-running bot sessions don't drift memory."""

    def __init__(self, cap: int = UNDO_CAP) -> None:
        self._stack: deque[UndoSnapshot] = deque(maxlen=cap)

    def push(self, snapshot: UndoSnapshot) -> None:
        self._stack.append(snapshot)

    def pop(self) -> UndoSnapshot | None:
        if not self._stack:
            return None
        return self._stack.pop()

    def __len__(self) -> int:
        return len(self._stack)
