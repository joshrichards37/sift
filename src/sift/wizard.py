"""Interactive setup wizard. Run via `sift-setup` (console script).

Walks the user through:
  1. Picking an LLM backend (Ollama / LM Studio / llama.cpp / MLX-LM / Hosted)
  2. Configuring that backend (varies — Ollama auto-pulls, others ask for
     URL + model and ping the endpoint to confirm it's up)
  3. Prompting for a Telegram bot token + auto-detecting the chat id
  4. Picking a preferences preset from examples/

No external TUI deps — plain stdin/stdout. Designed to fail loudly at any
step rather than silently writing a half-baked config.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
ENV_PATH = REPO_ROOT / ".env"
PREFS_PATH = REPO_ROOT / "preferences.yaml"


# Curated Ollama-tested presets. Tags are mapped from llmfit's recommendations
# to specific Ollama-library tags that are known to work.
@dataclass(frozen=True)
class ModelPreset:
    tag: str
    description: str
    vram_gb: float
    ram_gb: float
    tps_est: int


PRESETS: list[ModelPreset] = [
    ModelPreset(
        tag="qwen3:30b-a3b-instruct-2507-q4_K_M",
        description="Qwen3-30B-A3B-Instruct-2507 Q4_K_M — MoE, ~3B active. Best quality.",
        vram_gb=6,
        ram_gb=16,
        tps_est=95,
    ),
    ModelPreset(
        tag="qwen3:8b-q4_K_M",
        description="Qwen3-8B Q4_K_M — fits entirely in 6GB VRAM. Fast, decent quality.",
        vram_gb=6,
        ram_gb=8,
        tps_est=150,
    ),
    ModelPreset(
        tag="llama3.2:3b-instruct-q4_K_M",
        description="Llama 3.2 3B Instruct Q4_K_M — small, very fast. Acceptable summaries.",
        vram_gb=3,
        ram_gb=4,
        tps_est=200,
    ),
]


@dataclass(frozen=True)
class SysInfo:
    system: str  # 'Darwin', 'Linux', 'Windows'
    machine: str  # 'arm64', 'x86_64', etc.
    is_apple_silicon: bool


@dataclass(frozen=True)
class BackendConfig:
    base_url: str
    model: str
    api_key: str


# Hosted-API presets surfaced in the wizard. (label, base_url, default_model).
# `base_url=None` is the "Other / I'll type my own URL" escape hatch.
HOSTED_PRESETS: list[tuple[str, str | None, str | None]] = [
    ("OpenRouter (free + paid)", "https://openrouter.ai/api/v1", "google/gemini-2.5-flash"),
    ("Groq (free tier, fast)", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    (
        "Together AI",
        "https://api.together.xyz/v1",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ),
    ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
    ("Other (provide URL)", None, None),
]


def main() -> int:
    print_header()
    try:
        sys_info = detect_os()
        backend_key = pick_backend(sys_info)
        cfg = setup_backend(backend_key, sys_info)

        token = prompt_telegram_token()
        chat_id = asyncio.run(detect_chat_id(token))
        print(f"  ✓ chat id detected: {chat_id}\n")

        preset = pick_preset()

        write_env(token, chat_id, cfg)
        copy_preset(preset)

        print()
        print("=" * 60)
        print("Setup complete.")
        print()
        print(f"  .env              → {ENV_PATH}")
        print(f"  preferences.yaml  → {PREFS_PATH}  (from {preset.name})")
        print()
        print("Run the agent:")
        print("  uv run sift")
        print()
        print("In Telegram, DM your bot /start to confirm it's wired up.")
        return 0
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130
    except WizardError as e:
        print(f"\n✗ {e}")
        return 1


# ── Steps ────────────────────────────────────────────────────────────────


def print_header() -> None:
    print()
    print("sift setup wizard")
    print("=" * 60)
    print(
        "Walks you through LLM backend, Telegram bot, and preferences\n"
        "preset selection. Press Ctrl+C any time to abort.\n"
    )


def detect_os() -> SysInfo:
    s = platform.system()
    m = platform.machine()
    return SysInfo(
        system=s,
        machine=m,
        is_apple_silicon=(s == "Darwin" and m == "arm64"),
    )


def pick_backend(sys_info: SysInfo) -> str:
    print("[1/4] Choose LLM backend\n")
    options: list[tuple[str, str, str]] = [
        (
            "ollama",
            "Ollama",
            "easiest path. Cross-platform, auto-pulls models, recommended for first-timers.",
        ),
        ("lmstudio", "LM Studio", "GUI app, model browser. Especially nice on macOS."),
        (
            "llamacpp",
            "llama.cpp (llama-server)",
            "lean, GGUF-native. Run any Hugging Face GGUF.",
        ),
    ]
    if sys_info.is_apple_silicon:
        options.append(
            (
                "mlx",
                "MLX-LM",
                "native Apple Silicon — fastest on M-series. Browse mlx-community on HF.",
            )
        )
    options.append(
        (
            "hosted",
            "Hosted API",
            "OpenRouter / Groq / Together / OpenAI. Free tiers available; no local GPU needed.",
        )
    )

    for i, (_, label, blurb) in enumerate(options, 1):
        print(f"  {i}. {label}")
        print(f"     {blurb}")
    print()
    raw = input(f"Pick [1-{len(options)}, default 1]: ").strip() or "1"
    try:
        idx = int(raw)
    except ValueError:
        idx = 1
    if not (1 <= idx <= len(options)):
        idx = 1
    chosen = options[idx - 1]
    print(f"  ✓ {chosen[1]}\n")
    return chosen[0]


def setup_backend(key: str, sys_info: SysInfo) -> BackendConfig:
    if key == "ollama":
        return setup_ollama()
    if key == "lmstudio":
        return setup_lmstudio(sys_info)
    if key == "llamacpp":
        return setup_llamacpp(sys_info)
    if key == "mlx":
        return setup_mlx()
    if key == "hosted":
        return setup_hosted()
    raise WizardError(f"unknown backend: {key}")


# ── Ollama backend ───────────────────────────────────────────────────────


def setup_ollama() -> BackendConfig:
    print("[2/4] Ollama setup")
    check_ollama()
    hw = read_hardware()
    if hw:
        print_hardware(hw)
    model = pick_model(hw)
    pull_model(model)
    return BackendConfig(
        base_url="http://localhost:11434/v1",
        model=model.tag,
        api_key="ollama",
    )


def check_ollama() -> None:
    print("  Checking Ollama daemon…")
    if shutil.which("ollama") is None:
        raise WizardError(
            "Ollama is not installed.\n"
            "  macOS:  brew install ollama  (or download https://ollama.com)\n"
            "  Linux:  curl -fsSL https://ollama.com/install.sh | sh\n"
            "Then re-run this wizard."
        )
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        r.raise_for_status()
    except Exception as e:
        raise WizardError(
            f"Ollama is installed but not running ({e}).\n"
            "Start it with `ollama serve` (or as a systemd service) and try again."
        ) from e
    print("  ✓ Ollama running at localhost:11434\n")


def read_hardware() -> dict | None:
    """Best-effort hardware read via llmfit. Returns None if llmfit isn't installed."""
    if shutil.which("llmfit") is None:
        return None
    try:
        out = subprocess.run(
            ["llmfit", "system", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        try:
            return json.loads(out.stdout)
        except json.JSONDecodeError:
            pass
        rec = subprocess.run(
            ["llmfit", "recommend", "--json", "--limit", "1"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        data = json.loads(rec.stdout)
        return data.get("system")
    except Exception:
        return None


def print_hardware(hw: dict) -> None:
    print("  Hardware (via llmfit):")
    cpu = hw.get("cpu_name", "?")
    ram = hw.get("total_ram_gb", 0)
    backend = hw.get("backend", "CPU")
    gpus = hw.get("gpus", []) or []
    gpu_summary = "no GPU"
    if gpus:
        g = gpus[0]
        gpu_summary = f"{g.get('name', '?')} ({g.get('vram_gb', 0):.1f} GB VRAM, {backend})"
    print(f"    CPU:  {cpu}")
    print(f"    RAM:  {ram:.1f} GB")
    print(f"    GPU:  {gpu_summary}\n")


def pick_model(hw: dict | None) -> ModelPreset:
    print("  Choose a model:")
    candidates = _filter_to_hardware(PRESETS, hw)
    if not candidates:
        candidates = PRESETS  # let user pick anyway; they may know better
    for i, m in enumerate(candidates, 1):
        print(f"    {i}. {m.description}")
        print(
            f"         tag: {m.tag} · est. {m.tps_est} tok/s "
            f"· needs ~{m.vram_gb:.0f}GB VRAM, {m.ram_gb:.0f}GB RAM"
        )
    print()
    default = 1
    raw = input(f"  Pick [1-{len(candidates)}, default {default}]: ").strip() or str(default)
    try:
        idx = max(1, min(len(candidates), int(raw)))
    except ValueError:
        idx = default
    chosen = candidates[idx - 1]
    print(f"  ✓ {chosen.tag}\n")
    return chosen


def _filter_to_hardware(presets: list[ModelPreset], hw: dict | None) -> list[ModelPreset]:
    if not hw:
        return list(presets)
    ram = float(hw.get("total_ram_gb", 0) or 0)
    gpus = hw.get("gpus", []) or []
    vram = float(gpus[0].get("vram_gb", 0)) if gpus else 0.0
    return [p for p in presets if p.vram_gb <= vram + 0.5 and p.ram_gb <= ram + 0.5]


def pull_model(model: ModelPreset) -> None:
    print(f"  Pulling {model.tag} via Ollama (may take several minutes)…")
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        installed = {m["name"] for m in r.json().get("models", [])}
    except Exception:
        installed = set()
    if model.tag in installed:
        print("  ✓ already installed\n")
        return
    rc = subprocess.call(["ollama", "pull", model.tag])
    if rc != 0:
        raise WizardError(
            f"`ollama pull {model.tag}` failed (exit {rc}). Check the tag exists "
            f"on https://ollama.com/library and try again."
        )
    print()


# ── LM Studio backend ────────────────────────────────────────────────────


def setup_lmstudio(sys_info: SysInfo) -> BackendConfig:
    print("[2/4] LM Studio setup")
    print(
        "  Open LM Studio (download from https://lmstudio.ai if you haven't),\n"
        "  pull a model from the in-app Discover tab, then click the\n"
        "  'Developer' (or 'Local Server') tab and hit 'Start Server'.\n"
        "  Default endpoint is localhost:1234.\n"
    )
    base_url = prompt_url("LM Studio URL", "http://localhost:1234/v1")
    _verify_openai_endpoint(base_url, label="LM Studio")
    model = prompt_str(
        "Model identifier (visible in LM Studio under the loaded model)",
        "",
        required=True,
    )
    return BackendConfig(base_url=base_url, model=model, api_key="lmstudio")


# ── llama.cpp backend ────────────────────────────────────────────────────


def setup_llamacpp(sys_info: SysInfo) -> BackendConfig:
    print("[2/4] llama.cpp setup")
    install_hint = {
        "Darwin": "  Install: brew install llama.cpp",
        "Linux": "  Install: see https://github.com/ggml-org/llama.cpp#building",
    }.get(sys_info.system, "  Install: see https://github.com/ggml-org/llama.cpp")
    print(
        "  llama.cpp's `llama-server` binary speaks OpenAI-compatible /v1.\n"
        f"\n{install_hint}\n"
        "  Download a GGUF (e.g. https://huggingface.co/unsloth or\n"
        "  https://huggingface.co/bartowski) then run:\n"
        "    llama-server -m <path-to.gguf> -c 8192 --port 8080\n"
    )
    base_url = prompt_url("llama-server URL", "http://localhost:8080/v1")
    _verify_openai_endpoint(base_url, label="llama-server")
    model = prompt_str(
        "Model name (any string — llama-server reports whatever you pass)",
        "default",
    )
    return BackendConfig(base_url=base_url, model=model, api_key="llamacpp")


# ── MLX-LM backend (Apple Silicon) ───────────────────────────────────────


def setup_mlx() -> BackendConfig:
    print("[2/4] MLX-LM setup (Apple Silicon)")
    print(
        "  MLX-LM is the native Apple Silicon path — fastest on M-series.\n"
        "  Install + serve a model in one terminal:\n"
        "    pip install mlx-lm\n"
        "    mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080\n"
        "  Browse models at https://huggingface.co/mlx-community\n"
    )
    base_url = prompt_url("MLX-LM server URL", "http://localhost:8080/v1")
    _verify_openai_endpoint(base_url, label="mlx_lm.server")
    model = prompt_str(
        "HF model id you're serving",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
    )
    return BackendConfig(base_url=base_url, model=model, api_key="mlx")


# ── Hosted API backend ───────────────────────────────────────────────────


def setup_hosted() -> BackendConfig:
    print("[2/4] Hosted API setup")
    print("  Pick a provider:\n")
    for i, (label, _, _) in enumerate(HOSTED_PRESETS, 1):
        print(f"  {i}. {label}")
    print()
    raw = input(f"Pick [1-{len(HOSTED_PRESETS)}, default 1]: ").strip() or "1"
    try:
        idx = int(raw)
    except ValueError:
        idx = 1
    if not (1 <= idx <= len(HOSTED_PRESETS)):
        idx = 1
    label, url, default_model = HOSTED_PRESETS[idx - 1]
    print(f"  ✓ {label}\n")

    if url is None:
        url = prompt_url("Provider base URL (must end in /v1)", "")

    api_key = input("API key (won't be echoed back): ").strip()
    if not api_key:
        raise WizardError("API key is required for hosted backends.")

    model = prompt_str("Model name", default_model or "", required=True)

    return BackendConfig(base_url=url, model=model, api_key=api_key)


# ── Telegram + preferences ───────────────────────────────────────────────


def prompt_telegram_token() -> str:
    print("[3/4] Telegram bot")
    print(
        "  In Telegram, DM @BotFather and send /newbot. After picking a name\n"
        "  and username, BotFather will give you a token like 12345:AAH...\n"
    )
    while True:
        token = input("Paste the bot token: ").strip()
        if ":" not in token or len(token) < 30:
            print("  That doesn't look like a valid token; try again.")
            continue
        try:
            r = httpx.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                print(f"  Telegram rejected the token: {data}")
                continue
            username = data["result"]["username"]
            print(f"  ✓ token validates: @{username}\n")
            return token
        except httpx.HTTPError as e:
            print(f"  Telegram check failed ({e}); try again.")


async def detect_chat_id(token: str) -> int:
    print("  Now DM your bot in Telegram. Send any message (e.g. /start).")
    print("  Waiting up to 5 minutes for your message…\n")

    base = f"https://api.telegram.org/bot{token}"
    offset = 0
    deadline = asyncio.get_event_loop().time() + 300  # 5 minutes

    async with httpx.AsyncClient(timeout=30) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await client.get(
                    f"{base}/getUpdates",
                    params={"offset": offset, "timeout": 10},
                )
                r.raise_for_status()
                payload = r.json()
            except httpx.HTTPError as e:
                print(f"  ! Telegram poll error ({e}), retrying…")
                await asyncio.sleep(2)
                continue
            for update in payload.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message") or update.get("edited_message")
                if not msg:
                    continue
                chat = msg.get("chat", {})
                if chat.get("type") == "private":
                    return int(chat["id"])
        raise WizardError("Timed out waiting for a message. Re-run the wizard and try again.")


def pick_preset() -> Path:
    print("[4/4] Pick a preferences preset:")
    presets = sorted(EXAMPLES_DIR.glob("preferences-*.yaml"))
    if not presets:
        raise WizardError(f"No presets found in {EXAMPLES_DIR}")
    for i, p in enumerate(presets, 1):
        name = p.stem.removeprefix("preferences-")
        print(f"  {i}. {name}")
    print(f"  {len(presets) + 1}. blank skeleton — edit by hand")
    print()
    raw = input(f"Pick [1-{len(presets) + 1}, default 1]: ").strip() or "1"
    try:
        idx = int(raw)
    except ValueError:
        idx = 1
    if 1 <= idx <= len(presets):
        chosen = presets[idx - 1]
    else:
        chosen = REPO_ROOT / "preferences.example.yaml"
    print(f"  ✓ {chosen.name}\n")
    return chosen


# ── File output ──────────────────────────────────────────────────────────


def write_env(token: str, chat_id: int, cfg: BackendConfig) -> None:
    if ENV_PATH.exists():
        backup = ENV_PATH.with_suffix(".env.bak")
        ENV_PATH.replace(backup)
        print(f"  (backed up existing .env → {backup.name})")
    ENV_PATH.write_text(
        f"TELEGRAM_BOT_TOKEN={token}\n"
        f"OWNER_CHAT_ID={chat_id}\n"
        f"# AUTHORIZED_CHAT_IDS=\n"
        f"\n"
        f"LLM_BASE_URL={cfg.base_url}\n"
        f"LLM_MODEL={cfg.model}\n"
        f"LLM_API_KEY={cfg.api_key}\n"
        f"\n"
        f"# BLUESKY_HANDLE=\n"
        f"# BLUESKY_APP_PASSWORD=\n"
        f"\n"
        f"DB_PATH=./sift.db\n"
        f"PREFERENCES_PATH=./preferences.yaml\n"
    )
    os.chmod(ENV_PATH, 0o600)


def copy_preset(src: Path) -> None:
    if PREFS_PATH.exists():
        backup = PREFS_PATH.with_suffix(".yaml.bak")
        PREFS_PATH.replace(backup)
        print(f"  (backed up existing preferences.yaml → {backup.name})")
    shutil.copy(src, PREFS_PATH)


# ── Helpers ──────────────────────────────────────────────────────────────


def prompt_url(label: str, default: str) -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"  {label}{suffix}: ").strip() or default
    if not raw.startswith(("http://", "https://")):
        raise WizardError(f"URL must start with http:// or https:// — got {raw!r}")
    return raw


def prompt_str(label: str, default: str, *, required: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"  {label}{suffix}: ").strip()
    value = raw or default
    if required and not value:
        raise WizardError(f"{label} is required.")
    return value


def _verify_openai_endpoint(base_url: str, *, label: str = "endpoint") -> None:
    """GET /models to confirm the server is reachable. 401 still counts as
    'up' — means the server's running, just rejecting unauthenticated reads."""
    url = base_url.rstrip("/") + "/models"
    try:
        r = httpx.get(url, timeout=5.0)
        if r.status_code in (200, 401):
            print(f"  ✓ {label} reachable at {base_url}\n")
            return
        print(f"  ! {label} returned HTTP {r.status_code} from {url} — continuing anyway.\n")
    except Exception as e:
        print(f"  ! Couldn't reach {label} at {base_url}: {e}")
        print("    Continuing — sift will fail at runtime if this stays unreachable.\n")


class WizardError(RuntimeError):
    pass


if __name__ == "__main__":
    sys.exit(main())
