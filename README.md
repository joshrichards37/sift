# sift

> Self-hosted personal news agent. You pick the topics and sources; a local LLM filters the firehose down to what's relevant; a daily Telegram digest delivers it.

<p align="center">
  <img src="docs/img/digest-example.png" alt="A daily digest in Telegram showing five top articles with relevance scores and summaries" width="380">
</p>

## Why

Can't keep up with the fields you actually care about? Same. So I built this: a bot that watches your trusted sources, scores each article against your stated interests with a local LLM, and ships one Telegram digest a day. One message instead of forty open tabs. Runs on your own machine, your data stays yours.

Polls RSS, Hacker News, Bluesky (and anything else you write a Source for), scores each article 1-10 against your stated topics, summarises what passes, and sends one batched message a day. You can `/digest` for an immediate batch, `/more` to pull from the backlog, or just chat with the bot to ask follow-up questions over recent articles.

Single Python process, single SQLite file. Runs on a workstation. Optional small allowlist for sharing one feed with friends or family.

## Prereqs

- **Python 3.12** + [`uv`](https://docs.astral.sh/uv/)
- **An LLM backend** — anything OpenAI-compatible. The wizard sets up one of these for you:

  | | macOS | Linux | Notes |
  |---|---|---|---|
  | **[Ollama](https://ollama.com)** | ✓ | ✓ | recommended default — auto-pulls models |
  | **[LM Studio](https://lmstudio.ai)** | ✓ | ✓ | GUI app, especially nice on macOS |
  | **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (`llama-server`) | ✓ | ✓ | run any HF GGUF |
  | **[MLX-LM](https://github.com/ml-explore/mlx-lm)** | Apple Silicon only | — | native + fastest on M-series |
  | **Hosted API** (OpenRouter / Groq / Together / OpenAI / …) | ✓ | ✓ | no GPU required; great for VPS |

  Full setup details per backend in [`docs/backends.md`](docs/backends.md).

- **A Telegram bot** — DM [@BotFather](https://t.me/BotFather), `/newbot`, get a token (the wizard will validate it).
- (Optional) **[`llmfit`](https://github.com/disruptor-labs/llmfit)** — used by the Ollama branch of the wizard to read your hardware and recommend a model. Other backends don't need it.

## Setup — guided wizard

```bash
git clone https://github.com/joshrichards37/sift.git && cd sift
uv sync
uv run sift-setup
```

The wizard will:

1. Pick an LLM backend (Ollama / LM Studio / llama.cpp / MLX-LM / hosted)
2. Configure that backend (auto-pull for Ollama; URL + model prompt for the rest, with `/v1/models` reachability check)
3. Validate your Telegram bot token
4. Auto-detect your chat id (you DM the bot once, it reads the chat id from the message)
5. Pick a preferences preset from `examples/`
6. Write `.env` and `preferences.yaml`

Then start the agent:

```bash
uv run sift
```

DM your bot `/start` to confirm. First daily digest fires at the configured `digest_time` (default 09:00 local); `/digest` triggers one immediately.

## Setup — manual

If you'd rather skip the wizard:

```bash
git clone https://github.com/joshrichards37/sift.git && cd sift
uv sync
cp .env.example .env                                # fill in TELEGRAM_BOT_TOKEN + OWNER_CHAT_ID + LLM_*
cp examples/preferences-tech-news.yaml preferences.yaml   # or pick another preset
# bring up your LLM backend — see docs/backends.md
uv run sift
```

## Presets

`examples/` ships with ready-to-use topic configurations. Copy whichever fits your interests:

| Preset | Focus |
|---|---|
| `preferences-ai-tooling.yaml` | Harness engineering, Claude Code, agent SDKs, top AI-engineering voices |
| `preferences-tech-news.yaml` | Broad tech industry — hardware, software, security, dev tools |
| `preferences-research-papers.yaml` | arXiv (cs.CL/LG/AI) + lab blogs (HuggingFace, BAIR, Lilian Weng) |
| `preferences-finance-markets.yaml` | Macro, markets, investing — analytical-blog biased (free-source only) |

Each preset is a complete `preferences.yaml` — a topics block, exclusion keywords, threshold, and curated sources. Edit anything you like; the LLM scores articles against whatever you write in `topics`.

## Commands (in Telegram)

The agent runs in **digest mode**: articles are ingested + scored + summarised continuously, but only delivered as one batched message per day (default 09:00 local) plus on-demand via `/digest` and `/more`.

| Command | What it does |
|---|---|
| `/start` | Show status + command list |
| `/digest` | Send today's digest now (top N scored articles, marks them sent) |
| `/more [N]` | Send next N from backlog (default `more_size`, max 20) |
| `/backlog` | How many scored articles are queued for the next digest |
| `/prefs` | Threshold, digest size + time, source count, current backlog |
| `/pause` | Stop outbound messages (scheduler keeps ingesting + scoring) |
| `/resume` | Re-enable outbound |
| `/recent` | List last 10 sent articles |
| Any text | Free chat — model answers using context from last 20 sent articles |

## Inviting a friend

The bot supports a small allowlist of chat IDs sharing one feed (same preferences, same backlog). To add someone:

1. **They DM your bot first** — Telegram won't let the bot message anyone who hasn't initiated contact. Send them the bot username (e.g. `@ai_sift_bot`).
2. **They send any message** — `/start` is fine. The bot replies *"You're not authorised. Send your chat id `<their-actual-numeric-id>` to the owner."* The bot fills in their real id (e.g. `876543210`); they copy that number and forward it to you.
3. **You whitelist them** — append the id to `AUTHORIZED_CHAT_IDS` in `.env` (comma-separated), then restart.
4. **They DM `/start` again** — they're in. Daily digest, `/digest`, `/more` will all reach them.

What's shared vs. private:
- **Shared (everyone sees the same)**: the daily digest, `/digest`, `/more`. One backlog: if A `/more`s 10 articles, those 10 are gone for B until next digest.
- **Private (caller-only)**: `/backlog`, `/recent`, `/prefs`, free-text chat replies.
- **Global state, any user can flip**: `/pause` and `/resume` affect outbound for everyone.

Heads-up for friends:
- Bot runs on your machine. When your laptop sleeps, the bot is offline.
- Free-text chat queries flow through your Ollama (you can read them in the DB if you want — they should know).

If your group grows past ~5 active users, consider having them self-host instead (clone the repo, register their own BotFather token, run their own Ollama or use Gemini Flash free tier). Same code, no infra burden on you.

## Adding a source

1. New file in `src/sift/sources/<name>.py` subclassing `Source`.
2. Implement `async def poll() -> list[Article]`. Idempotent — caller dedups by URL.
3. Wire into the factory in `src/sift/sources/__init__.py` keyed by source-id prefix.
4. Document in `preferences.example.yaml` under `sources:`.

## Tuning

| Symptom | Knob |
|---|---|
| Telegram floods | raise `relevance_threshold`, lower `max_per_cycle`, lengthen `cadence_seconds` |
| Missing relevant items | lower threshold, sharpen `topics` to be more specific |
| Bad summaries | edit `SUMMARY_SYSTEM` in `src/sift/llm.py`, restart |
| Too slow | switch to a smaller model (`LLM_MODEL=qwen3:8b`) and re-pull |

## Switching the LLM

Any OpenAI-compatible endpoint works. Edit the three `LLM_*` lines in `.env` and restart:

```bash
# Gemini Flash via OpenRouter (free tier)
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=google/gemini-2.5-flash
```

Full per-backend instructions (Ollama, LM Studio, llama.cpp, MLX-LM, OpenRouter, Groq, Together, OpenAI) in [`docs/backends.md`](docs/backends.md). Pre-existing `OLLAMA_*` env vars also still work — they're aliased for back-compat.

## Storage

SQLite at `./sift.db`. Schema in `src/sift/storage.py` (`articles`, `feedback`, `source_cursor`). Inspect with `sqlite3 sift.db`.

## Lint / format

```bash
uv run ruff check src/
uv run ruff format src/
```

No tests yet — added when there's a regression worth pinning.

## Sample systemd user unit

Drop into `~/.config/systemd/user/sift.service`:

```ini
[Unit]
Description=sift personal news agent
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/workspace/sift
ExecStart=%h/.local/bin/uv run sift
Restart=on-failure
RestartSec=30s

[Install]
WantedBy=default.target
```

Then `systemctl --user daemon-reload && systemctl --user enable --now sift`.

## Documentation

- [`docs/backends.md`](docs/backends.md) — Ollama / LM Studio / llama.cpp / MLX-LM / hosted APIs (OpenRouter, Groq, Together, OpenAI)
- [`docs/models.md`](docs/models.md) — picking a model within a backend; quality / speed tradeoffs
- [`docs/sources.md`](docs/sources.md) — source types, cadence guidelines, writing your own
- [`docs/prompting.md`](docs/prompting.md) — how relevance + summary prompts work, tuning
- [`docs/deploy.md`](docs/deploy.md) — tmux, systemd, launchd, VPS patterns
- [`CLAUDE.md`](CLAUDE.md) — architecture, conventions, contributing context for AI agents
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — PR process, style, commit conventions

## License

MIT. See [`LICENSE`](LICENSE).
