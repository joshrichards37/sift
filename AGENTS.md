# AGENTS.md

This file provides guidance for AI agents (Claude Code, Codex, Cursor, OpenHands, Aider, etc.) working in this repo. It's the canonical agent-instructions file; `CLAUDE.md` is a symlink for tools that look for that name specifically.

## Repository purpose

`sift` is **a self-hosted personal news agent**. Polls feeds the operator cares about (RSS, Hacker News, Reddit, Bluesky, etc.), filters by user-stated topics via a local LLM, and delivers a single daily digest to Telegram. Two-way: the operator chats with the bot to ask follow-up questions over recent articles, adjust preferences, or thumbs-up/down for relevance feedback.

Designed to run unattended on a workstation or small VPS — single Python process, single SQLite file, no Docker required. Supports a small allowlist of additional chat IDs sharing one feed (friends/family with the same topic interests).

## Stack

- **Python 3.12** managed by `uv` (PEP 735 dev deps under `[dependency-groups]`)
- **`python-telegram-bot[job-queue]`** — bot interface and async runtime
- **LLM via OpenAI-compatible client** — defaults to local Ollama at `http://localhost:11434`, but the wizard supports LM Studio, llama.cpp's `llama-server`, MLX-LM, and hosted APIs (OpenRouter / Groq / Together / OpenAI). Model swap = env-var change, no code edit. See `docs/backends.md`.
- **SQLite** (stdlib `sqlite3`) for articles + dedup + feedback. Single-writer process, no concurrency concerns.
- **`feedparser`** for RSS, **`atproto`** for Bluesky, plain `httpx` for HN Algolia and Reddit JSON.

## Structure

```
/
├── pyproject.toml              uv-managed; deps + ruff + pytest config
├── preferences.example.yaml    topic-agnostic skeleton (placeholders)
├── examples/                   ready-to-use presets (ai-tooling, tech-news,
│                               research-papers, finance-markets)
├── .env.example                TELEGRAM_BOT_TOKEN, OWNER_CHAT_ID,
│                               AUTHORIZED_CHAT_IDS, LLM_*, BLUESKY_*
├── src/sift/
│   ├── main.py                 entrypoint — wires bot + scheduler + digest
│   │                           loop under one asyncio gather
│   ├── wizard.py               sift-setup CLI: backend picker (ollama,
│   │                           lm-studio, llama.cpp, mlx, hosted), model
│   │                           pick + pull, telegram token + chat-id
│   │                           auto-detect, preset pick, write .env + prefs
│   ├── config.py               Pydantic Settings (env) + YAML preferences
│   ├── storage.py              SQLite schema + helpers
│   ├── llm.py                  OpenAI-compat client; relevance + summarize
│   ├── digest.py               daily digest scheduler + /digest /more logic
│   ├── sources/
│   │   ├── base.py             Source ABC: poll() → list[Article]
│   │   ├── rss.py              feedparser-backed
│   │   ├── hn.py               Algolia API; splits "OR" client-side
│   │   ├── reddit.py           public JSON API; min_points score filter
│   │   └── bluesky.py          atproto author feed
│   ├── scheduler.py            asyncio loop; per-source cadence + jitter
│   └── telegram_bot.py         handlers + send_message_safe broadcast
├── tests/                      pytest suite (config, sources, llm, storage,
│                               telegram chunking)
├── .github/workflows/ci.yml    ruff + pytest on PRs to main
└── docs/                       backends, models, sources, prompting, deploy
```

## Build, test, run

```bash
uv sync                              # install deps
uv run sift-setup                    # interactive first-time setup
uv run sift                          # run the agent
uv run ruff check src/               # lint
uv run ruff format src/              # format
uv run pytest                        # run unit tests
```

Whichever LLM backend you picked must be running before `uv run sift`. The wizard verifies it during setup; if you swap backends manually in `.env`, see `docs/backends.md`.

## Quality gates

Before committing:

```bash
uv run ruff check src/
uv run ruff format --check src/
uv run pytest
```

CI runs these on every PR (`.github/workflows/ci.yml`); the `lint` check is required by branch protection on `main`.

## Architecture notes

- **One process, three asyncio tasks.** The Telegram bot's `Application` runs in the loop alongside (a) the source scheduler — `asyncio.gather` of per-source poller tasks, each `sleep(cadence + jitter)` then `poll()` — and (b) the digest loop, which sleeps until the next `digest_time` and emits a batched message.
- **Digest mode, not live push.** The scheduler scores + summarises continuously and stashes results in the DB; nothing goes to Telegram on its own. The daily digest at `prefs.digest_time` (default 09:00 local) selects top-N unpushed scored articles ≥ threshold, sends them as one message, and marks them `pushed_at`. `/digest` and `/more` are the on-demand triggers.
- **Articles are immutable.** Once ingested, the article row is only mutated for `relevance_score`/`summary` (set by the scoring step) and `pushed_at` (set when delivered). Feedback (thumbs) is a separate table keyed by article id.
- **Dedup is by URL hash, not LLM.** Cheap, deterministic, runs before the LLM ever sees the article. The LLM's job is relevance scoring and summarization, not "is this a duplicate."
- **Relevance is a single LLM call per article**, returns a 1–10 score against the user's `preferences.yaml`. Threshold (default 7) is configurable. Below threshold = drop silently; at/above = summarize and queue for the next digest.
- **`max_per_cycle` is a *scoring* cap, not a push cap.** Limits LLM calls per source per poll so a backlog doesn't burn the GPU on first run. Articles beyond the cap are deferred to the next poll cycle.
- **Multi-user is a flat allowlist sharing one feed.** `OWNER_CHAT_ID` + comma-separated `AUTHORIZED_CHAT_IDS` in `.env`. Daily digest broadcasts to every chat. Per-user preferences / per-user backlog is YAGNI — if a friend wants their own knobs, they self-host.
- **Summary prompt is templated**, pinned in `llm.py`. Don't inline prompts in business logic — they need to be diffable when tuning.
- **The LLM client is OpenAI-compatible by design.** Ollama, LM Studio, llama.cpp's `llama-server`, MLX-LM, vLLM, OpenRouter, Groq, Gemini-via-OpenRouter, etc. all expose `/v1/chat/completions`. Model swap = env var change, no code edit.

## Conventions

- **Source IDs are stable strings** (`hn`, `rss:slug`, `reddit:sub`, `bsky:handle`, `github:slug`, `arxiv:slug`, `masto:slug`). Used as foreign keys in storage and filter labels in preferences. Never reuse an ID for a different source. Allowed kinds live in `sift.sources.KNOWN_KINDS`; `SourcePref._check_kind` validates at parse time.
- **Times are UTC, ISO-8601 strings in the DB**, `datetime` objects in code. Convert at the boundary only.
- **Secrets in `.env`, preferences in `preferences.yaml`.** Both are gitignored. `.env.example` and `preferences.example.yaml` ship in the repo as starting templates; `examples/` ships full domain presets.
- **No Telegram message > 4096 chars.** `_chunk()` in `telegram_bot.py` splits on paragraph boundaries to keep HTML well-formed across chunks.
- **Imports**: stdlib → third-party → first-party, separated by blank lines. Ruff's `I` rule enforces.

## Adding a new source

1. New file in `src/sift/sources/<name>.py` subclassing `Source`.
2. Implement `async def poll() -> list[Article]`. Idempotent — caller dedups by URL.
3. Register in `sources/__init__.py` factory keyed by source-id prefix (e.g. `kind == "youtube"`).
4. Document in `preferences.example.yaml` under `sources:` with default cadence.
5. Add a small parsing test in `tests/test_sources.py` exercising at least one realistic payload edge case.

## Hardware + model notes

When the wizard's Ollama backend branch is selected, `llmfit recommend --json` (optional [llmfit](https://github.com/disruptor-labs/llmfit) CLI) reads CPU/RAM/GPU and ranks models. The wizard maps the recommendation to one of three Ollama-tested presets in `wizard.py:PRESETS`:

- **Qwen3-30B-A3B-Instruct-2507 Q4_K_M** — MoE, ~3B active. Best quality. Needs ~6GB VRAM (active experts) + ~16GB RAM (offloaded experts). ~95 tok/s on a mid-range laptop GPU.
- **Qwen3-8B Q4_K_M** — fits entirely in 6GB VRAM. ~150 tok/s. Slightly worse at nuanced relevance.
- **Llama 3.2 3B Instruct Q4_K_M** — small, very fast (~200 tok/s). Acceptable summaries, weaker filtering.

Re-run the wizard or edit `LLM_MODEL` in `.env` to swap. Other backends (LM Studio, llama.cpp, MLX-LM, hosted APIs) bring their own model selection — see `docs/backends.md`.

## Git workflow

Open PRs against `main`. Branch protection requires 1 approving review and a passing `lint` check (CI runs ruff + pytest). Force-push to `main` is blocked.

## Don't add

- **Tests for everything.** Add tests when (a) a bug recurs, (b) a refactor needs a safety net, or (c) an LLM-prompt regression would be expensive. A personal news bot doesn't need 80% coverage. The current suite covers env-var resolution, source payload parsing, LLM JSON handling, storage roundtrips, and chunking — high-leverage spots.
- **Migrations framework.** SQLite + a single `init_db()` that runs `CREATE TABLE IF NOT EXISTS`. If the schema needs to change incompatibly, write a one-off Python script, don't pull in Alembic.
- **Web UI / dashboard.** Telegram is the UI. If you reach for Flask, stop and ask why.
- **Per-user preferences / multi-tenant abstraction.** Flat allowlist with shared feed is intentional. Friends who want their own knobs self-host.
- **Cloud deploy / Docker as the default path.** Runs on a workstation. The README points out a $10/mo VPS path for those who want it, but the default is local-first.

## Writing style

Concise and detailed. No fluff. Opinionated, explicit *why*, minimal hand-holding.

## Comments + commits

- Default to no comments. Add one only when the *why* is non-obvious (a Telegram API quirk, a feedparser bug workaround, a prompt fragment that's load-bearing). Never comment the *what*.
- Commit messages: lowercase, imperative, scope-prefixed (`feat(sources): add bluesky firehose`, `fix(llm): retry on 503`). Body explains the why, not the how.

## Operating environment notes (for agent harnesses)

Some shells alias `cp` / `mv` / `rm` to interactive mode (`-i`), which causes agents to hang waiting for y/n input. Always use non-interactive flags:

```bash
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

Other commands that may prompt:

- `scp` / `ssh` — `-o BatchMode=yes`
- `apt-get` — `-y`
- `brew` — `HOMEBREW_NO_AUTO_UPDATE=1`
- `git rebase`, `git add` — never use the `-i` flag (interactive mode is unsupported)
