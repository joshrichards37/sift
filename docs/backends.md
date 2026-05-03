# Backends

sift's LLM client speaks the OpenAI-compatible Chat Completions API (`POST /v1/chat/completions`). Anything that exposes that endpoint can serve sift — local runtimes, hosted APIs, your own vLLM cluster.

The setup wizard (`uv run sift-setup`) walks you through the five most common options. This doc covers each in detail if you want to set one up manually or pick which to use.

## At a glance

| Backend | Install | Speed | Cost | Best for |
|---|---|---|---|---|
| **Ollama** | one command | fast | free | first-timers; cross-platform default |
| **LM Studio** | GUI download | fast | free | macOS users who like a GUI |
| **llama.cpp (`llama-server`)** | build/brew | very fast | free | any HF GGUF; lowest overhead |
| **MLX-LM** | `pip install` | very fast | free | Apple Silicon — native, fastest on M-series |
| **Hosted API** | API key | very fast | free tier or pay | no GPU; VPS deploys; mobile workstation |

All five share the same three env vars in `.env`. Switching backends = editing those three lines, no code changes:

```bash
LLM_BASE_URL=...    # /v1 endpoint
LLM_MODEL=...       # model name the backend reports
LLM_API_KEY=...     # any non-empty string for local backends; real key for hosted
```

(Pre-existing `OLLAMA_*` env vars also still work — they're aliased for back-compat.)

---

## Ollama

Default backend. Cross-platform, auto-pulls models, single-binary daemon. The wizard's happy path.

### Install

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Then run the daemon (or set it up as a service):

```bash
ollama serve
```

### Pull + configure

```bash
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
```

```bash
# .env
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M
LLM_API_KEY=ollama   # any non-empty string; Ollama doesn't check
```

See [`docs/models.md`](models.md) for which tag to pick for your hardware.

---

## LM Studio

GUI app with a built-in model browser and one-click local server. Mac-first but cross-platform.

### Install

Download from <https://lmstudio.ai> (macOS, Linux AppImage, Windows installer).

### Run a model

1. **Discover** tab → search (e.g. "Qwen2.5 7B Instruct GGUF") → download
2. **Developer** (or **Local Server**, version-dependent) tab → **Start Server**

The server defaults to `localhost:1234` and exposes `/v1/chat/completions`.

### Configure

```bash
# .env
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=qwen2.5-7b-instruct   # the identifier LM Studio shows for the loaded model
LLM_API_KEY=lmstudio            # any non-empty string
```

**Notes:** cleanest path for macOS users who don't want CLI. Supports MLX models on Apple Silicon (faster than GGUF on M-series).

---

## llama.cpp (`llama-server`)

Lean, GGUF-native. Best when you're already pulling models from Hugging Face manually and want minimal overhead.

### Install

```bash
# macOS
brew install llama.cpp

# Linux / Windows: build from source
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && make
```

### Run a model

```bash
# download a GGUF — unsloth and bartowski both publish well-quantized variants
huggingface-cli download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --local-dir ~/models

# serve it
llama-server -m ~/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  -c 8192 --port 8080
```

### Configure

```bash
# .env
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=default               # llama-server reports any string you want
LLM_API_KEY=llamacpp            # any non-empty string
```

**Notes:** often the fastest local runtime for a given GGUF, especially with `-ngl <n>` to offload N layers to GPU. See `llama-server --help` for context length, parallel decoding, and other flags.

---

## MLX-LM (Apple Silicon only)

Native M-series support via Apple's MLX framework. Usually beats GGUF on the same chip.

### Install + serve

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
```

### Configure

```bash
# .env
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit
LLM_API_KEY=mlx                 # any non-empty string
```

**Notes:**
- Browse compatible models at <https://huggingface.co/mlx-community>.
- M1 / M2 / M3 / M4 only. Intel Macs need Ollama or llama.cpp.
- Comparable to LM Studio's MLX backend but lighter (no GUI).

---

## Hosted APIs

No local GPU required. Right call for VPS deploys, weak laptops, or when you don't want a long-running daemon. All speak OpenAI-compat.

### OpenRouter

Aggregator. Free-tier daily quota on several models.

```bash
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...           # https://openrouter.ai/keys
LLM_MODEL=google/gemini-2.5-flash
```

### Groq

Fastest hosted inference (300+ tok/s). Free tier handles one user comfortably.

```bash
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...             # https://console.groq.com
LLM_MODEL=llama-3.3-70b-versatile
```

### Together AI

Pay-as-you-go, wide model selection.

```bash
LLM_BASE_URL=https://api.together.xyz/v1
LLM_API_KEY=...                 # https://api.together.xyz/settings/api-keys
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
```

### OpenAI

```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini           # cheap + fast; gpt-4o for higher quality
```

### Anything else OpenAI-compatible

vLLM, TGI, LiteLLM, OpenLLM, llama-cpp-python's `--api-server` mode, an internal company gateway. Just point `LLM_BASE_URL` at the `/v1` route.

---

## Switching backends later

Two paths:

1. **Re-run the wizard:** `uv run sift-setup` (it backs up your existing `.env`).
2. **Edit `.env` by hand:** change the three `LLM_*` lines and restart `uv run sift`.

---

## Validating a backend (`sift-bench`)

Before committing a new backend to a long unattended run, benchmark it. `sift-bench` runs a fixed set of synthetic articles through the configured backend's scoring + summary paths, captures per-call latency and token usage, and projects per-user/day cost given the standard sift workload (100 articles scored to deliver 10).

```bash
# Basic run — uses your existing .env config
uv run sift-bench

# Smaller / faster
uv run sift-bench --n-scoring 10 --n-summary 3

# With cost projection (rates as of 2026-05; check current pricing)
uv run sift-bench --input-cost 0.59 --output-cost 0.79      # Groq Llama 3.3 70B
uv run sift-bench --input-cost 0.15 --output-cost 0.60      # OpenAI gpt-4o-mini
uv run sift-bench --input-cost 0.10 --output-cost 0.40      # Gemini 2.5 Flash via OpenRouter

# Machine-readable output for scripting / CI
uv run sift-bench --json
```

Sample output:

```
sift-bench — backend validation
================================
Backend:                    https://api.groq.com/openai/v1
Model:                      llama-3.3-70b-versatile
Calls:                      30 scoring, 10 summary

Latency
-------
  Scoring  p50 / p95:            420 ms /      650 ms
  Summary  p50 / p95:            900 ms /     1300 ms

Token usage (per call, average)
--------------------------------
  Scoring  in / out:             520    /       45
  Summary  in / out:            1480    /      135

Projection — 1 user / day  (100 scored, 10 summarised)
-----------------------------------------------------------------
  Tokens in / out:               66,800 /      5,850
  Wall time:                     51.0 s
  Cost / user / day:        $0.0440
```

What to look for:

- **Wall time per user/day** — if this exceeds a few seconds for one user, scaling is going to hurt. Self-hosted backends should clock in well under a minute; hosted backends typically under 30s.
- **p95 vs p50 latency** — large gaps signal occasional rate limiting or cold-starts. Investigate before going unattended.
- **Cost / user / day × your user count** — multiply by the number of authorised chats sharing the feed. Hosted backends with per-user economics should be well under $1/user/month at this workload.
- **Failed JSON parsing** — score returns `parse-failed` when the model produces invalid JSON. Common with reasoning-tuned variants (`-thinking`, `o1`-like, `r1`). Switch to a chat-tuned model.

The benchmark is sequential (one call at a time) — measures single-stream latency, which matches sift's per-source scheduler. It does not stress-test concurrent throughput.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Connection refused" on startup | Backend daemon not running. Start it. |
| "401 Unauthorized" | Wrong / missing `LLM_API_KEY`. Hosted endpoints require a real key. |
| Garbage scores or `parse-failed` in logs | Model isn't honouring the JSON-output instruction. Use a chat-tuned model — avoid `-thinking`, `-coder`, `o1`-like, `deepseek-r1`, or other reasoning-tuned variants. |
| Very slow | Smaller model, or move to a hosted backend. See [`docs/models.md`](models.md). |
