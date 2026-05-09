"""Interactive setup wizard. Run via `sift-setup` (console script).

Walks the user through three stages — LLM backend, Telegram bot, and
preferences. Every stage can be skipped; skipped stages write
`__TODO_*__` placeholders into `.env` that `sift` startup detects and
surfaces with the resume command. Re-run `sift-setup --resume <stage>`
to fill them in.

Uses questionary for arrow-key menus, password fields, and graceful
Ctrl+C handling. The wizard is the only place questionary is imported
— the agent itself runs headless.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import questionary

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
ENV_PATH = REPO_ROOT / ".env"
PREFS_PATH = REPO_ROOT / "preferences.yaml"

# Sentinel placed in `.env` for fields the user skipped. Startup greps for
# this prefix and bails with the resume command rather than letting Pydantic
# emit a confusing validation traceback.
TODO_PREFIX = "__TODO_"
SKIP = "__SKIP__"

STAGES = ("backend", "telegram", "prefs")


@dataclass(frozen=True)
class ModelPreset:
    tag: str
    description: str
    vram_gb: float
    ram_gb: float
    tps_est: int


# Curated Ollama-tested presets. Replaced with a live picker in news-stream-zj7;
# kept here as the offline fallback when the Ollama daemon isn't reachable.
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
    system: str
    machine: str
    is_apple_silicon: bool


@dataclass
class BackendConfig:
    base_url: str
    model: str
    api_key: str


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


@dataclass
class WizardState:
    sys_info: SysInfo
    backend: BackendConfig | None = None
    telegram_token: str | None = None
    chat_id: int | None = None
    preset_path: Path | None = None
    skipped: list[str] = field(default_factory=list)


class WizardError(RuntimeError):
    pass


# ── Entry ────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()
    print_header(args.resume)
    try:
        state = WizardState(sys_info=detect_os())
        existing = parse_env(ENV_PATH) if ENV_PATH.exists() else {}
        stages = _stages_to_run(args.resume)

        if "backend" in stages:
            run_backend_stage(state)
        else:
            preserve_backend(state, existing)

        if "telegram" in stages:
            run_telegram_stage(state)
        else:
            preserve_telegram(state, existing)

        if "prefs" in stages:
            run_prefs_stage(state)

        write_env(state)
        if "prefs" in stages and state.preset_path is not None:
            copy_preset(state.preset_path)

        print_summary(state)
        return 0
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130
    except WizardError as e:
        print(f"\n✗ {e}")
        return 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sift-setup", description="sift setup wizard")
    p.add_argument(
        "--resume",
        choices=("all", *STAGES),
        default="all",
        help="run only one stage and merge into existing .env (default: all)",
    )
    return p.parse_args()


def _stages_to_run(resume: str) -> set[str]:
    return set(STAGES) if resume == "all" else {resume}


# ── Stages ───────────────────────────────────────────────────────────────


def run_backend_stage(state: WizardState) -> None:
    print_step(1, "LLM backend")
    backend_key = pick_backend(state.sys_info)
    if backend_key == SKIP:
        state.skipped.append("backend")
        print("  (skipped — fill in LLM_* in .env later or re-run `sift-setup --resume backend`)\n")
        return
    state.backend = setup_backend(backend_key, state.sys_info)


def run_telegram_stage(state: WizardState) -> None:
    print_step(2, "Telegram bot")
    if not _confirm_run_stage("Configure Telegram bot now?"):
        state.skipped.append("telegram")
        print(
            "  (skipped — fill TELEGRAM_BOT_TOKEN + OWNER_CHAT_ID in .env, "
            "or re-run `sift-setup --resume telegram`)\n"
        )
        return
    state.telegram_token = prompt_telegram_token()
    state.chat_id = asyncio.run(detect_chat_id(state.telegram_token))
    print(f"  ✓ chat id detected: {state.chat_id}\n")


def run_prefs_stage(state: WizardState) -> None:
    print_step(3, "Preferences")
    if not _confirm_run_stage("Pick a preferences preset now?"):
        state.skipped.append("prefs")
        print(
            "  (skipped — copy one of examples/preferences-*.yaml to "
            "preferences.yaml later, or re-run `sift-setup --resume prefs`)\n"
        )
        return
    state.preset_path = pick_preset()


# ── Backend picker ───────────────────────────────────────────────────────


def pick_backend(sys_info: SysInfo) -> str:
    choices: list[questionary.Choice] = [
        questionary.Choice(
            "Ollama  — easiest path. Cross-platform, auto-pulls models.",
            value="ollama",
        ),
        questionary.Choice(
            "LM Studio  — GUI app, model browser. Especially nice on macOS.",
            value="lmstudio",
        ),
        questionary.Choice(
            "llama.cpp (llama-server)  — lean, GGUF-native.",
            value="llamacpp",
        ),
    ]
    if sys_info.is_apple_silicon:
        choices.append(
            questionary.Choice(
                "MLX-LM  — native Apple Silicon, fastest on M-series.",
                value="mlx",
            )
        )
    choices += [
        questionary.Choice(
            "Hosted API  — OpenRouter / Groq / Together / OpenAI.",
            value="hosted",
        ),
        questionary.Separator(),
        questionary.Choice("Skip — configure later", value=SKIP),
    ]
    answer = questionary.select("Choose LLM backend", choices=choices, default=choices[0]).ask()
    if answer is None:
        raise KeyboardInterrupt
    if answer != SKIP:
        label = next(
            c.title for c in choices if isinstance(c, questionary.Choice) and c.value == answer
        )
        print(f"  ✓ {label.split('  —')[0].strip()}\n")
    return answer


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


# ── Ollama ───────────────────────────────────────────────────────────────


def setup_ollama() -> BackendConfig:
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
    candidates = _filter_to_hardware(PRESETS, hw)
    if not candidates:
        candidates = PRESETS
    choices = [
        questionary.Choice(
            f"{m.description}\n         "
            f"tag: {m.tag} · est. {m.tps_est} tok/s · "
            f"~{m.vram_gb:.0f}GB VRAM, {m.ram_gb:.0f}GB RAM",
            value=m,
        )
        for m in candidates
    ]
    answer = questionary.select("Choose a model", choices=choices, default=choices[0]).ask()
    if answer is None:
        raise KeyboardInterrupt
    print(f"  ✓ {answer.tag}\n")
    return answer


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


# ── LM Studio / llama.cpp / MLX-LM ───────────────────────────────────────


def setup_lmstudio(sys_info: SysInfo) -> BackendConfig:
    print(
        "  Open LM Studio (download from https://lmstudio.ai if you haven't),\n"
        "  pull a model from the Discover tab, then under 'Developer' (or\n"
        "  'Local Server') hit 'Start Server'. Default endpoint is :1234.\n"
    )
    base_url = prompt_url("LM Studio URL", "http://localhost:1234/v1")
    _verify_openai_endpoint(base_url, label="LM Studio")
    model = prompt_str(
        "Model identifier (visible in LM Studio under the loaded model)",
        "",
        required=True,
    )
    return BackendConfig(base_url=base_url, model=model, api_key="lmstudio")


def setup_llamacpp(sys_info: SysInfo) -> BackendConfig:
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


def setup_mlx() -> BackendConfig:
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


# ── Hosted ───────────────────────────────────────────────────────────────


def setup_hosted() -> BackendConfig:
    choices = [
        questionary.Choice(label, value=(label, url, model)) for label, url, model in HOSTED_PRESETS
    ]
    answer = questionary.select("Pick a provider", choices=choices, default=choices[0]).ask()
    if answer is None:
        raise KeyboardInterrupt
    label, url, default_model = answer
    print(f"  ✓ {label}\n")

    if url is None:
        url = prompt_url("Provider base URL (must end in /v1)", "")

    api_key = questionary.password("API key").ask()
    if api_key is None:
        raise KeyboardInterrupt
    api_key = api_key.strip()
    if not api_key:
        raise WizardError("API key is required for hosted backends.")

    model = prompt_str("Model name", default_model or "", required=True)
    return BackendConfig(base_url=url, model=model, api_key=api_key)


# ── Telegram ─────────────────────────────────────────────────────────────


def prompt_telegram_token() -> str:
    print(
        "  In Telegram, DM @BotFather and send /newbot. After picking a name\n"
        "  and username, BotFather will give you a token like 12345:AAH...\n"
    )
    while True:
        token = questionary.password("Paste the bot token").ask()
        if token is None:
            raise KeyboardInterrupt
        token = token.strip()
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
    deadline = asyncio.get_event_loop().time() + 300

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


# ── Preferences ──────────────────────────────────────────────────────────


def pick_preset() -> Path:
    presets = sorted(EXAMPLES_DIR.glob("preferences-*.yaml"))
    if not presets:
        raise WizardError(f"No presets found in {EXAMPLES_DIR}")
    choices = [questionary.Choice(p.stem.removeprefix("preferences-"), value=p) for p in presets]
    blank = REPO_ROOT / "preferences.example.yaml"
    choices.append(questionary.Choice("blank skeleton — edit by hand", value=blank))
    answer = questionary.select(
        "Pick a preferences preset", choices=choices, default=choices[0]
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    print(f"  ✓ {answer.name}\n")
    return answer


# ── File output ──────────────────────────────────────────────────────────


def parse_env(path: Path) -> dict[str, str]:
    """Parse a `.env`-style file into a dict. Comment lines and blanks are
    skipped; quoted values have their quotes stripped. Used by --resume to
    preserve fields the user isn't re-running."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        out[k.strip()] = v
    return out


def preserve_backend(state: WizardState, env: dict[str, str]) -> None:
    """Populate state.backend from existing .env. Skipped silently if any
    field is missing or carries a TODO marker — write_env will re-stub it."""
    base_url = env.get("LLM_BASE_URL", "")
    model = env.get("LLM_MODEL", "")
    api_key = env.get("LLM_API_KEY", "")
    if all(_is_real(v) for v in (base_url, model, api_key)):
        state.backend = BackendConfig(base_url=base_url, model=model, api_key=api_key)
    else:
        state.skipped.append("backend")


def preserve_telegram(state: WizardState, env: dict[str, str]) -> None:
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    chat_id_raw = env.get("OWNER_CHAT_ID", "")
    if _is_real(token) and _is_real(chat_id_raw):
        try:
            state.chat_id = int(chat_id_raw)
            state.telegram_token = token
        except ValueError:
            state.skipped.append("telegram")
    else:
        state.skipped.append("telegram")


def _is_real(v: str) -> bool:
    return bool(v) and not v.startswith(TODO_PREFIX)


def write_env(state: WizardState) -> None:
    """Write `.env` from `state`. Missing fields become `__TODO_<NAME>__`
    placeholders that `sift` startup detects and surfaces. Existing `.env`
    is backed up to `.env.bak`."""
    if ENV_PATH.exists():
        backup = ENV_PATH.with_suffix(".env.bak")
        ENV_PATH.replace(backup)
        print(f"  (backed up existing .env → {backup.name})")

    token = state.telegram_token or _todo("TELEGRAM_BOT_TOKEN")
    chat_id = str(state.chat_id) if state.chat_id is not None else _todo("OWNER_CHAT_ID")
    base_url = state.backend.base_url if state.backend else _todo("LLM_BASE_URL")
    model = state.backend.model if state.backend else _todo("LLM_MODEL")
    api_key = state.backend.api_key if state.backend else _todo("LLM_API_KEY")

    ENV_PATH.write_text(
        f"TELEGRAM_BOT_TOKEN={token}\n"
        f"OWNER_CHAT_ID={chat_id}\n"
        f"# AUTHORIZED_CHAT_IDS=\n"
        f"\n"
        f"LLM_BASE_URL={base_url}\n"
        f"LLM_MODEL={model}\n"
        f"LLM_API_KEY={api_key}\n"
        f"\n"
        f"# BLUESKY_HANDLE=\n"
        f"# BLUESKY_APP_PASSWORD=\n"
        f"\n"
        f"DB_PATH=./sift.db\n"
        f"PREFERENCES_PATH=./preferences.yaml\n"
    )
    os.chmod(ENV_PATH, 0o600)


def _todo(name: str) -> str:
    return f"{TODO_PREFIX}{name}__"


def copy_preset(src: Path) -> None:
    if PREFS_PATH.exists():
        backup = PREFS_PATH.with_suffix(".yaml.bak")
        PREFS_PATH.replace(backup)
        print(f"  (backed up existing preferences.yaml → {backup.name})")
    shutil.copy(src, PREFS_PATH)


# ── Helpers ──────────────────────────────────────────────────────────────


def prompt_url(label: str, default: str) -> str:
    raw = questionary.text(label, default=default).ask()
    if raw is None:
        raise KeyboardInterrupt
    raw = raw.strip()
    if not raw.startswith(("http://", "https://")):
        raise WizardError(f"URL must start with http:// or https:// — got {raw!r}")
    return raw


def prompt_str(label: str, default: str, *, required: bool = False) -> str:
    raw = questionary.text(label, default=default).ask()
    if raw is None:
        raise KeyboardInterrupt
    value = raw.strip() or default
    if required and not value:
        raise WizardError(f"{label} is required.")
    return value


def _confirm_run_stage(prompt: str) -> bool:
    """Y/N prompt for skipping a stage. Defaults to yes (run it) since users
    arriving at the wizard generally want to fill things in."""
    answer = questionary.confirm(prompt, default=True).ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


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


# ── Output ───────────────────────────────────────────────────────────────


def print_header(resume: str) -> None:
    print()
    print("sift setup wizard")
    print("=" * 60)
    if resume == "all":
        print(
            "Three stages: LLM backend, Telegram bot, preferences.\n"
            "Skip any stage and re-run with `--resume <stage>` later.\n"
        )
    else:
        print(f"Resuming stage: {resume}\n")


def print_step(idx: int, label: str) -> None:
    print(f"[{idx}/3] {label}\n")


def detect_os() -> SysInfo:
    s = platform.system()
    m = platform.machine()
    return SysInfo(
        system=s,
        machine=m,
        is_apple_silicon=(s == "Darwin" and m == "arm64"),
    )


def print_summary(state: WizardState) -> None:
    print()
    print("=" * 60)
    print("Setup complete." if not state.skipped else "Setup partially complete.")
    print()
    print(f"  .env              → {ENV_PATH}")
    if state.preset_path is not None:
        print(f"  preferences.yaml  → {PREFS_PATH}  (from {state.preset_path.name})")
    elif "prefs" in state.skipped:
        print("  preferences.yaml  → not written (stage skipped)")
    print()
    if state.skipped:
        print("Skipped stages — fill in later:")
        for stage in state.skipped:
            print(f"  · {stage}      uv run sift-setup --resume {stage}")
        print()
        print("`uv run sift` will refuse to start until every TODO marker is filled in.")
    else:
        print("Run the agent:")
        print("  uv run sift")
        print()
        print("In Telegram, DM your bot /start to confirm it's wired up.")


# ── Public helpers consumed by main.py ───────────────────────────────────


def find_todo_stubs(env_path: Path) -> list[str]:
    """Return the list of `.env` keys whose value is still a `__TODO_*__`
    placeholder. Empty list = setup is complete enough to start the agent."""
    if not env_path.exists():
        return ["__no_env_file__"]
    out: list[str] = []
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if v.strip().startswith(TODO_PREFIX):
            out.append(k.strip())
    return out


if __name__ == "__main__":
    sys.exit(main())
