from __future__ import annotations

from pathlib import Path

import pytest

from sift.config import Preferences, Settings, SourcePref, load_preferences

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    """Build a Settings without touching the on-disk .env file."""
    for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "OLLAMA_URL", "OLLAMA_MODEL", "OLLAMA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test:token")
    monkeypatch.setenv("OWNER_CHAT_ID", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return Settings(_env_file=None)


def test_settings_canonical_llm_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(
        monkeypatch,
        LLM_BASE_URL="http://canonical/v1",
        LLM_MODEL="m-canon",
        LLM_API_KEY="k-canon",
    )
    assert s.llm_base_url == "http://canonical/v1"
    assert s.llm_model == "m-canon"
    assert s.llm_api_key == "k-canon"


def test_settings_ollama_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(
        monkeypatch,
        OLLAMA_URL="http://legacy/v1",
        OLLAMA_MODEL="m-legacy",
        OLLAMA_API_KEY="k-legacy",
    )
    assert s.llm_base_url == "http://legacy/v1"
    assert s.llm_model == "m-legacy"
    assert s.llm_api_key == "k-legacy"


def test_settings_canonical_wins_over_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both LLM_* and OLLAMA_* set: canonical wins. Critical: someone with an
    old .env that already has OLLAMA_URL must be able to override with LLM_URL
    without removing the legacy line."""
    s = _settings(
        monkeypatch,
        LLM_BASE_URL="http://winner",
        OLLAMA_URL="http://loser",
    )
    assert s.llm_base_url == "http://winner"


def test_chat_ids_owner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, AUTHORIZED_CHAT_IDS="")
    assert s.chat_ids == [1]


def test_chat_ids_with_friends(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, AUTHORIZED_CHAT_IDS="2,3, 4 ,5")
    # Owner first, friends in declared order, whitespace tolerated
    assert s.chat_ids == [1, 2, 3, 4, 5]


def test_chat_ids_dedup_owner_in_friends_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """If owner_id is also in AUTHORIZED_CHAT_IDS, don't double up."""
    s = _settings(monkeypatch, AUTHORIZED_CHAT_IDS="1,2,3")
    assert s.chat_ids == [1, 2, 3]


def test_chat_ids_invalid_entries_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, AUTHORIZED_CHAT_IDS="2,not-a-number,3,")
    assert s.chat_ids == [1, 2, 3]


@pytest.mark.parametrize(
    "preset",
    sorted(EXAMPLES.glob("preferences-*.yaml")),
    ids=lambda p: p.name,
)
def test_each_shipped_preset_parses(preset: Path) -> None:
    """Every preset under examples/ must parse to a valid Preferences object.
    Catches typos and stale schema fields when presets get edited."""
    prefs = load_preferences(preset)
    assert isinstance(prefs, Preferences)
    assert prefs.topics  # non-empty
    assert prefs.sources  # at least one source declared
    # All source IDs unique within a preset (storage uses them as foreign keys)
    ids = [s.id for s in prefs.sources]
    assert len(ids) == len(set(ids)), f"duplicate source ids in {preset.name}: {ids}"


def test_source_pref_accepts_known_kinds() -> None:
    """All four supported kinds — and the kind:slug variants — should parse."""
    for valid_id in [
        "hn",
        "hn:finance",
        "rss:simon-willison",
        "reddit:rust",
        "bsky:simonw",
    ]:
        SourcePref(id=valid_id)  # would raise on invalid


def test_source_pref_rejects_unknown_kind() -> None:
    """The exact bug that motivated this validator: 'hn-finance' (hyphen instead
    of colon) must fail at parse time with a message that names the bad kind
    AND the valid set, so the fix is obvious."""
    with pytest.raises(ValueError) as exc_info:
        SourcePref(id="hn-finance")
    msg = str(exc_info.value)
    assert "hn-finance" in msg
    assert "hn:finance" in msg or "hn:<slug>" in msg or "kind:slug" in msg.lower() or "hn" in msg
    # Valid kinds enumerated so the user knows what's allowed
    for kind in ("hn", "rss", "reddit", "bsky"):
        assert kind in msg


def test_load_preferences_surfaces_bad_source_id_at_parse_time(tmp_path: Path) -> None:
    """End-to-end: a preferences.yaml with a typo'd source id fails at
    load_preferences(), not at startup in build_sources. This is the UX win
    — the user sees the problem when they save the file, not minutes later."""
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(
        """
topics: |
  whatever
sources:
  - id: hn-finance
    enabled: true
    query: "anything"
""".strip()
    )
    with pytest.raises(ValueError) as exc_info:
        load_preferences(yaml_path)
    assert "hn-finance" in str(exc_info.value)


def test_preferences_round_trip(tmp_path: Path) -> None:
    yaml = tmp_path / "p.yaml"
    yaml.write_text(
        """
topics: |
  test topic
exclude_keywords: [foo, bar]
relevance_threshold: 8
sources:
  - id: hn
    enabled: true
    query: "x OR y"
    min_points: 100
""".strip()
    )
    prefs = load_preferences(yaml)
    assert prefs.topics.strip() == "test topic"
    assert prefs.exclude_keywords == ["foo", "bar"]
    assert prefs.relevance_threshold == 8
    assert len(prefs.sources) == 1
    assert prefs.sources[0].query == "x OR y"
    assert prefs.sources[0].min_points == 100
