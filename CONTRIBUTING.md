# Contributing

PRs welcome. A few notes to make them likely to land.

## Running the project locally

```bash
uv sync
uv run sift-setup        # interactive — creates .env + preferences.yaml
uv run sift              # run the agent
```

For a quick edit-test loop without sending real Telegram messages: edit the relevant module, then in another terminal smoke-test against the existing DB:

```bash
uv run python -c "
import asyncio
from pathlib import Path
from sift.config import Settings, load_preferences
from sift.digest import run_digest
# ... call the function you're touching
"
```

## Style

- Python 3.12, `uv` for everything (`uv add`, `uv run`).
- Ruff handles lint + format. Before pushing:

  ```bash
  uv run ruff check src/
  uv run ruff format src/
  uv run pytest        # if tests exist for the area you're touching
  ```

- Default to no comments. Add a comment only when *why* is non-obvious — a workaround for an external API quirk, a load-bearing prompt fragment, an invariant that isn't visible from the code.
- Imports: stdlib → third-party → first-party, separated by blank lines (Ruff's `I` rule enforces).

## Commit messages

Lowercase, imperative, scope-prefixed:

- `feat(sources): add youtube channel source`
- `fix(telegram): fall back to plain text on parse error`
- `docs(readme): clarify wizard chat-id flow`
- `chore(deps): bump python-telegram-bot 21.6 → 21.9`

Body explains the *why*, not the *what*.

## What goes where

- **New source** (e.g. Mastodon, RSS aggregator with auth): subclass `Source` in `src/sift/sources/<name>.py`, register in `sources/__init__.py`'s factory keyed on a stable id prefix. Document in `preferences.example.yaml`.
- **New Telegram command**: handler method on `Bot`, register in `_wire_handlers`, add to `_start`'s help text and the README command table.
- **New preset**: drop a `preferences-<name>.yaml` file in `examples/`. Keep it ready-to-use — full topics block, full curated source list, threshold/cadence already set.
- **New LLM provider**: shouldn't need code changes if it's OpenAI-compatible (most are). If it's not, the `LLM` class in `llm.py` is the only thing to edit.

## What we don't want

- **Tests-for-everything** changes that don't pin down real regressions.
- **Web UI / dashboard / Docker compose** — Telegram is the UI; SQLite + a single Python process is the deployment model.
- **Per-user preferences / multi-tenant abstraction** — the flat allowlist is intentional. If you want personalised feeds, run your own instance.
- **Migration framework** — SQLite + `CREATE TABLE IF NOT EXISTS` in `init_db()` is enough. If you need an incompatible schema change, write a one-off Python script, don't pull in Alembic.
- **Dependencies for things stdlib already does** — we have nine deps total and want to keep it under fifteen.

## Reporting a bug

Issues that are easy to act on:

- Exact error message + stack trace.
- The relevant `preferences.yaml` snippet (redact secrets).
- Ollama version + model tag, or the OpenAI-compatible endpoint you're hitting.
- Whether it reproduces with one of the shipped `examples/` presets unmodified.

Issues that are hard to act on: "it doesn't work."

## Security

Don't open a public issue for anything that involves credentials, RCE, or exfiltration. Email the maintainer directly (see `pyproject.toml` author field) or DM via GitHub.
