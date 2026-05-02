# Models

This doc is about *which model* to pick for sift's workload. For *how to host* a model (Ollama / LM Studio / llama.cpp / MLX-LM / hosted APIs), see [`docs/backends.md`](backends.md).

The setup wizard's Ollama branch ships with three Ollama-tested presets. They're chosen for the work this bot does — short, structured prompts (relevance scoring) and ~80-word summaries — not for general chat.

| Model | VRAM | RAM | tok/s | Quality | When to pick |
|---|---:|---:|---:|---|---|
| **Qwen3-30B-A3B-Instruct-2507 Q4_K_M** | ~6GB (active experts) | ~16GB (offloaded experts) | ~95 | Strong | Default. MoE means 3B active params do the work; the rest sit in system RAM. Best relevance calibration. |
| **Qwen3-8B Q4_K_M** | ~6GB | 8GB | ~150 | Good | Laptop without 16GB free RAM, or you want the GPU exclusively. Slightly worse at edge-case relevance. |
| **Llama 3.2 3B Instruct Q4_K_M** | ~3GB | 4GB | ~200 | Acceptable | Old hardware, integrated GPU, or you're CPU-bound. Summaries are fine; relevance scoring is noisier. |

`uv run sift-setup` filters these to what your hardware can actually run (via [`llmfit`](https://github.com/disruptor-labs/llmfit)) and pre-picks the best. You can re-run the wizard or just edit `LLM_MODEL` in `.env` to swap.

## Using a different model

Anything Ollama hosts works. Pull it (`ollama pull <tag>`), set `LLM_MODEL=<tag>` in `.env`, restart the agent. Worth trying:

- `gpt-oss:20b` — Anthropic-released open-weights, decent for summaries.
- `mistral-small:24b-instruct-2509-q4_K_M` — strong instruction following.
- `phi4:14b-q4_K_M` — Microsoft's, good cost/quality balance.

Don't pick reasoning-tuned models (`qwen3-coder`, `o1`-likes, `deepseek-r1`) — they generate long chains of thought before the answer, which makes scoring 50× slower and the JSON parsing flakier.

## Using a hosted provider instead

The LLM client is OpenAI-compatible. Any provider with that API works:

```bash
# Gemini Flash via OpenRouter (free tier, 1500 req/day)
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=google/gemini-2.5-flash

# Groq (free tier, very fast)
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile

# Together AI
LLM_BASE_URL=https://api.together.xyz/v1
LLM_API_KEY=...
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
```

`LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` are the canonical names. Pre-existing `.env` files using `OLLAMA_*` still work — the code reads either.

## Cost notes

If you're polling 10 sources at 30-min cadence, expect ~30-100 LLM calls per hour at steady state (most sources return 0-2 new items per poll). Each call is a relevance score (~500 input tokens, ~30 output tokens) plus a summary on ones that pass (~3000 input, ~120 output).

- **Local Ollama**: free. GPU sits idle most of the time.
- **Gemini Flash free tier**: well under the 1500/day limit for one user.
- **Groq free tier**: same.
- **Paid APIs**: budget a couple of dollars a month if you go cloud, more if you raise `max_per_cycle` past 5 or shorten cadences below 15 min.
