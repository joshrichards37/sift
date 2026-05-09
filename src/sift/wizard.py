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
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import questionary
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
ENV_PATH = REPO_ROOT / ".env"
PREFS_PATH = REPO_ROOT / "preferences.yaml"

# Sentinel placed in `.env` for fields the user skipped. Startup greps for
# this prefix and bails with the resume command rather than letting Pydantic
# emit a confusing validation traceback.
TODO_PREFIX = "__TODO_"
SKIP = "__SKIP__"
CUSTOM_TAG = "__CUSTOM__"  # picker sentinel for "let me type any Ollama tag"

# Curated model manifest hosted in the repo. Fetched at wizard time so we can
# add or retire models without users having to upgrade `sift`. Falls back to
# bundled PRESETS on any network or parse error.
RECOMMENDATIONS_URL = (
    "https://raw.githubusercontent.com/joshrichards37/sift/main/data/ollama_recommendations.json"
)

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
    drafted_yaml: str | None = None  # set if user accepted an LLM-drafted prefs.yaml
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
        if "prefs" in stages:
            if state.drafted_yaml is not None:
                write_drafted_prefs(state.drafted_yaml)
            elif state.preset_path is not None:
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
    if not _confirm_run_stage("Configure preferences now?"):
        state.skipped.append("prefs")
        print(
            "  (skipped — copy one of examples/preferences-*.yaml to "
            "preferences.yaml later, or re-run `sift-setup --resume prefs`)\n"
        )
        return

    # Drafting needs a working backend; without one we can only offer presets.
    mode = _pick_prefs_mode(can_draft=state.backend is not None)
    if mode == "draft":
        drafted = _draft_prefs_loop(state)
        if drafted is not None:
            state.drafted_yaml = drafted
            return
        # User abandoned the draft — fall through to preset pick rather than
        # leave them with nothing.
        print("  Falling back to preset selection.\n")
    state.preset_path = pick_preset()


def _pick_prefs_mode(*, can_draft: bool) -> str:
    """Returns 'draft' or 'preset'. Skips the prompt entirely when drafting
    is unavailable (no backend) — there's only one path forward."""
    if not can_draft:
        return "preset"
    choices = [
        questionary.Choice(
            "Describe what you follow — let the LLM draft a preferences.yaml",
            value="draft",
        ),
        questionary.Choice(
            "Pick a preset from examples/  (ai-tooling, tech-news, …)",
            value="preset",
        ),
    ]
    answer = questionary.select(
        "How do you want to configure preferences?",
        choices=choices,
        default=choices[0],
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


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
    if hw and _hardware_is_useful(hw):
        print_hardware(hw)
    else:
        hw = None  # don't let zero-filled output drive the fit warning
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


def _hardware_is_useful(hw: dict) -> bool:
    """llmfit occasionally returns a stub dict (all zeros / unknowns) when it
    can't read the system. Print-and-warn off that data is worse than silent —
    treat empty hardware data as 'no info available' downstream."""
    cpu = hw.get("cpu_name") or ""
    ram = float(hw.get("total_ram_gb", 0) or 0)
    gpus = hw.get("gpus") or []
    return bool(cpu and cpu != "?") or ram > 0 or bool(gpus)


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
    """Three-tier picker: (1) already-installed Ollama models, (2) curated
    recommendations fetched from the sift repo (offline-fallback to bundled
    PRESETS), (3) free-form custom tag. Hardware fit is a warning, not a
    filter — the user may know their setup better than llmfit does."""
    installed = _get_installed_ollama_models()
    recs = _fetch_recommendations() or PRESETS
    rec_by_tag = {m.tag: m for m in recs}

    choices: list[questionary.Choice | questionary.Separator] = []
    if installed:
        choices.append(questionary.Separator("── installed ──"))
        for tag in installed:
            m = rec_by_tag.get(tag) or _bare_preset(tag)
            choices.append(questionary.Choice(_render_model_choice(m, hw, installed=True), value=m))

    fresh_recs = [m for m in recs if m.tag not in installed]
    if fresh_recs:
        choices.append(questionary.Separator("── recommended (will be pulled) ──"))
        for m in fresh_recs:
            label = _render_model_choice(m, hw, installed=False)
            choices.append(questionary.Choice(label, value=m))

    choices.append(questionary.Separator())
    choices.append(questionary.Choice("Type a custom Ollama tag…", value=CUSTOM_TAG))

    # questionary.Separator inherits from Choice, so isinstance(c, Choice) is
    # True for both — must filter Separators explicitly or default lookup
    # picks a non-selectable one and questionary raises.
    default = next(
        (
            c
            for c in choices
            if isinstance(c, questionary.Choice) and not isinstance(c, questionary.Separator)
        ),
        None,
    )
    answer = questionary.select("Choose a model", choices=choices, default=default).ask()
    if answer is None:
        raise KeyboardInterrupt
    if answer == CUSTOM_TAG:
        return _prompt_custom_tag()
    print(f"  ✓ {answer.tag}\n")
    return answer


def _render_model_choice(m: ModelPreset, hw: dict | None, *, installed: bool) -> str:
    """Format one model row for the picker. Installed models get a ✓ prefix;
    over-budget models (per llmfit hw read, if available) get a ⚠ suffix —
    we still show them, just signal the cost."""
    prefix = "✓ " if installed else "  "
    head = m.description or m.tag
    parts = [f"{prefix}{head}"]
    metadata: list[str] = []
    if m.tps_est:
        metadata.append(f"~{m.tps_est} tok/s")
    if m.vram_gb or m.ram_gb:
        metadata.append(f"~{m.vram_gb:.0f}GB VRAM, {m.ram_gb:.0f}GB RAM")
    if not installed:
        metadata.append(f"tag: {m.tag}")
    if metadata:
        parts.append("         " + " · ".join(metadata))
    if not installed and hw and not _fits_hardware(m, hw):
        parts.append("         ⚠ over your hardware budget — will offload to CPU/disk")
    return "\n".join(parts)


def _fits_hardware(m: ModelPreset, hw: dict | None) -> bool:
    if not hw or (not m.vram_gb and not m.ram_gb):
        return True
    ram = float(hw.get("total_ram_gb", 0) or 0)
    gpus = hw.get("gpus", []) or []
    vram = float(gpus[0].get("vram_gb", 0)) if gpus else 0.0
    return m.vram_gb <= vram + 0.5 and m.ram_gb <= ram + 0.5


def _get_installed_ollama_models() -> list[str]:
    """Best-effort list of currently-installed Ollama tags. Empty list on any
    error (daemon unreachable, parse failure) — the picker still works using
    just the recommendations + custom-tag escape hatch."""
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _fetch_recommendations() -> list[ModelPreset] | None:
    """Pull the curated manifest from RECOMMENDATIONS_URL. None on any failure
    — caller falls back to the bundled PRESETS list. Manifest format mirrors
    ModelPreset fields one-for-one."""
    try:
        r = httpx.get(RECOMMENDATIONS_URL, timeout=5.0)
        r.raise_for_status()
        data = r.json()
        return [
            ModelPreset(
                tag=item["tag"],
                description=item.get("description", ""),
                vram_gb=float(item.get("vram_gb", 0)),
                ram_gb=float(item.get("ram_gb", 0)),
                tps_est=int(item.get("tps_est", 0)),
            )
            for item in data
        ]
    except Exception:
        return None


def _bare_preset(tag: str) -> ModelPreset:
    """ModelPreset for an installed model we have no metadata for. The
    picker just shows the tag with no fit warning."""
    return ModelPreset(tag=tag, description="", vram_gb=0, ram_gb=0, tps_est=0)


def _prompt_custom_tag() -> ModelPreset:
    """Free-form tag entry. We can't cheaply pre-validate uninstalled tags
    (Ollama's registry has no public 'does this exist' endpoint), so we
    accept any non-empty input and let `pull_model` surface the error if
    the tag is bogus."""
    tag = questionary.text(
        "Ollama tag",
        default="",
        instruction="(e.g. qwen3:8b, mistral:7b-instruct, llama3.1:70b)",
    ).ask()
    if tag is None:
        raise KeyboardInterrupt
    tag = tag.strip()
    if not tag:
        raise WizardError("Tag cannot be empty.")
    print(f"  ✓ {tag} (will validate at pull time)\n")
    return _bare_preset(tag)


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


# ── LLM-drafted preferences ──────────────────────────────────────────────


# Pinned in the wizard rather than llm.py because the agent itself never
# generates preferences — only the wizard does. Keeping the prompt local
# means tuning it doesn't ripple into the runtime path.
DRAFT_PREFS_SYSTEM = """You generate a complete preferences.yaml for sift, a personal news bot. \
The user describes what they want to follow in plain language. You output \
ONLY the YAML — no preamble, no commentary, no markdown fences.

Required structure:

topics: |
  - <bullet 1: be specific, name people/products/concepts where you can>
  - <bullet 2>
  - <bullet 3>

  Score down: <kinds of articles to push back on, e.g. funding rounds w/o product implications>

exclude_keywords:
  - <hard-exclusion keyword if obviously off-topic>
relevance_threshold: 7
summary_target_words: 50
max_per_cycle: 3
digest_time: "09:00"
digest_size: 10
more_size: 5

sources:
  - id: <source id, see allowed kinds below>
    enabled: true
    cadence_seconds: 1800
    <kind-specific fields>

Allowed source kinds and their fields:
  - id: hn                    fields: query (str, multiple terms joined " OR "), min_points (int)
  - id: rss:<slug>            fields: url (str, full feed URL)
  - id: reddit:<slug>         fields: subreddit (str, single name or "a+b+c"), min_points (int)
  - id: bsky:<handle>         fields: handle (str, e.g. someone.bsky.social)

Generate 3-6 source entries matching the user's interests.

Hard constraints — silently ignored sources are worse than no sources:

- DO NOT include any bsky: entry unless the user explicitly mentions Bluesky, the AT \
  Protocol, or names a Bluesky handle. Bluesky sources require BLUESKY_HANDLE and \
  BLUESKY_APP_PASSWORD env vars that most users don't have set, so an unsolicited \
  bsky entry will raise on every poll.

- For reddit: only use subreddits you are highly confident exist. Common safe ones: \
  programming, MachineLearning, LocalLLaMA, rust, golang, python, datascience, \
  artificial, OpenAI, askscience, ChatGPT, learnmachinelearning. If you're tempted \
  to invent a subreddit name (e.g. r/AI, r/Tech, r/Coding), DO NOT — those don't \
  exist. Use a broader well-known sub or fall back to HN/RSS instead.

- HN queries: use double quotes around exact phrases, never single quotes (single \
  quotes get URL-encoded literally). Make sure every quote is balanced. \
  Example query string: \"large language model\" OR llm OR \"local AI\"

- RSS URLs must be plausible. If you don't know a real feed URL for a source, prefer \
  an HN keyword search over a guessed RSS URL.

Output the YAML and nothing else."""


def _draft_prefs_loop(state: WizardState) -> str | None:
    """Run the description → draft → review loop. Returns the accepted YAML
    string or None if the user gave up. State.backend must be populated."""
    assert state.backend is not None
    if _looks_small(state.backend.model):
        print(
            "  ⚠ The chosen model looks small (~8B or under). Drafted YAML may be\n"
            "    rough — feel free to re-prompt or fall back to a preset.\n"
        )
    description = _prompt_description(initial=None)
    if not description:
        print("  (empty description — cancelling draft)\n")
        return None

    while True:
        print(f"  Asking {state.backend.model} for a preferences.yaml…")
        try:
            yaml_text = asyncio.run(_call_llm_for_yaml(state.backend, description))
        except Exception as e:
            print(f"  ✗ LLM call failed: {e}\n")
            return None
        print()
        print("  ── Drafted preferences.yaml ──")
        print(_indent(yaml_text, "  "))
        print()
        valid = _yaml_is_valid(yaml_text)
        if not valid:
            print("  ⚠ This doesn't validate against the Preferences schema.\n")
        else:
            yaml_text = _maybe_disable_dead_sources(yaml_text)

        accept_disabled = "invalid YAML — edit first" if not valid else None
        action = questionary.select(
            "What now?",
            choices=[
                questionary.Choice("Accept and save", value="accept", disabled=accept_disabled),
                questionary.Choice("Re-prompt with a different description", value="reprompt"),
                questionary.Choice("Edit in $EDITOR before saving", value="edit"),
                questionary.Choice("Cancel — pick a preset instead", value="cancel"),
            ],
        ).ask()
        if action is None:
            raise KeyboardInterrupt
        if action == "accept":
            return yaml_text
        if action == "reprompt":
            new_desc = _prompt_description(initial=description)
            if new_desc:
                description = new_desc
            continue
        if action == "edit":
            edited = _edit_in_editor(yaml_text)
            if not edited.strip():
                print("  (empty after edit — keeping previous draft)\n")
                continue
            if not _yaml_is_valid(edited):
                print("  ⚠ Edited YAML still doesn't validate.")
                keep = questionary.confirm("Save invalid YAML anyway?", default=False).ask()
                if keep:
                    return edited
                yaml_text = edited
                continue
            return edited
        return None  # cancel


async def _call_llm_for_yaml(backend: BackendConfig, description: str) -> str:
    """Ask the user's configured LLM to draft a preferences.yaml. Returns the
    raw model output with code fences stripped. Validation happens in the
    caller — we don't want to retry-loop here when the model returns
    something unparseable."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=backend.base_url, api_key=backend.api_key)
    resp = await client.chat.completions.create(
        model=backend.model,
        messages=[
            {"role": "system", "content": DRAFT_PREFS_SYSTEM},
            {
                "role": "user",
                "content": f"User interests:\n\n{description}\n\nGenerate the YAML.",
            },
        ],
        temperature=0.4,
    )
    return _strip_code_fences(resp.choices[0].message.content or "")


def _strip_code_fences(s: str) -> str:
    """Strip ```yaml ... ``` fences if the model wrapped its output, even
    though we explicitly asked it not to. Smaller models forget this rule."""
    s = s.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
    if s.endswith("```"):
        s = s[: s.rfind("```")]
    return s.strip()


def _yaml_is_valid(yaml_text: str) -> bool:
    """Validate that the drafted YAML parses *and* matches the Preferences
    schema (so we don't accept output that has the right shape but breaks
    Pydantic on first run)."""
    if not yaml_text.strip():
        return False
    try:
        from sift.config import Preferences

        data = yaml.safe_load(yaml_text)
        Preferences.model_validate(data)
        return True
    except Exception:
        return False


def _maybe_disable_dead_sources(yaml_text: str) -> str:
    """Pre-flight every source in the drafted YAML and offer to auto-disable
    any that fail. Returns the YAML (possibly modified). Reasoning: drafted
    sources can be syntactically valid but operationally dead — hallucinated
    subreddits, bsky entries without creds, broken RSS URLs. Catching these
    before save spares the user a startup full of repeating tracebacks."""
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return yaml_text  # validity already checked upstream; trust the caller
    sources = data.get("sources") or []
    if not sources:
        return yaml_text

    print("  Pre-flight: checking each source…")
    failures: list[tuple[str, str]] = asyncio.run(_check_sources(sources))
    if not failures:
        print(f"  ✓ all {len(sources)} sources reachable\n")
        return yaml_text

    print(f"  ⚠ {len(failures)} of {len(sources)} sources look dead:")
    for src_id, reason in failures:
        print(f"    · {src_id}  — {reason}")
    print()

    answer = questionary.confirm("Auto-disable these (set enabled: false)?", default=True).ask()
    if answer is None:
        raise KeyboardInterrupt
    if not answer:
        print("  (keeping them enabled — sift will log errors until you fix them)\n")
        return yaml_text

    bad_ids = {src_id for src_id, _ in failures}
    for source in sources:
        if source.get("id") in bad_ids:
            source["enabled"] = False
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=120)


async def _check_sources(sources: list[dict]) -> list[tuple[str, str]]:
    """Run the per-kind reachability checks concurrently. Returns
    (id, reason) for every source that failed."""
    async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
        results = await asyncio.gather(
            *(_check_one_source(client, src) for src in sources),
            return_exceptions=True,
        )
    out: list[tuple[str, str]] = []
    for src, res in zip(sources, results, strict=True):
        if isinstance(res, str):  # error message
            out.append((src.get("id", "<unknown>"), res))
        elif isinstance(res, BaseException):
            out.append((src.get("id", "<unknown>"), f"check raised: {res}"))
    return out


async def _check_one_source(client: httpx.AsyncClient, source: dict) -> str | None:
    """None on success, or a one-line failure message. We don't want this
    check to block forever on a slow feed, so each call has a short timeout
    via the shared client."""
    src_id = source.get("id", "")
    kind = src_id.split(":", 1)[0]

    if kind == "hn":
        return None  # Algolia is always reachable; query validity is content not connectivity

    if kind == "rss":
        url = source.get("url") or ""
        if not url:
            return "missing url"
        try:
            r = await client.get(url)
            if r.status_code >= 400:
                return f"HTTP {r.status_code}"
            if not r.content:
                return "empty response"
            return None
        except Exception as e:
            return f"unreachable ({type(e).__name__})"

    if kind == "reddit":
        sub = source.get("subreddit") or ""
        if not sub:
            return "missing subreddit"
        # /about.json returns 404 when a sub doesn't exist, 403 if private/quarantined.
        check_url = f"https://www.reddit.com/r/{sub.split('+')[0]}/about.json"
        try:
            r = await client.get(check_url, headers={"User-Agent": "sift-setup/1.0"})
            if r.status_code == 404:
                return "subreddit not found"
            if r.status_code == 403:
                return "subreddit is private or quarantined"
            if r.status_code >= 400:
                return f"HTTP {r.status_code}"
            return None
        except Exception as e:
            return f"check failed ({type(e).__name__})"

    if kind == "bsky":
        if not _bluesky_creds_present():
            return "BLUESKY_HANDLE/BLUESKY_APP_PASSWORD not set in .env"
        handle = source.get("handle") or ""
        if not handle:
            return "missing handle"
        try:
            r = await client.get(
                "https://bsky.social/xrpc/com.atproto.identity.resolveHandle",
                params={"handle": handle},
            )
            if r.status_code >= 400:
                return f"handle resolve failed ({r.status_code})"
            return None
        except Exception as e:
            return f"check failed ({type(e).__name__})"

    return f"unknown source kind: {kind!r}"


def _bluesky_creds_present() -> bool:
    """Check the on-disk .env for BLUESKY_* — we can't rely on os.environ
    because Pydantic Settings reads .env at startup, not into the env."""
    env = parse_env(ENV_PATH)
    return _is_real(env.get("BLUESKY_HANDLE", "")) and _is_real(env.get("BLUESKY_APP_PASSWORD", ""))


def _looks_small(model_tag: str) -> bool:
    """Heuristic: does the model tag look like a model with ≤8B parameters?
    Used only to surface a soft warning — false negatives (e.g. MoE models
    where the digit is total params, not active) are tolerable since the
    user can ignore the warning anyway."""
    m = re.search(r"[:\-_](\d+(?:\.\d+)?)b\b", model_tag.lower())
    if not m:
        return False
    return float(m.group(1)) <= 8


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _prompt_description(*, initial: str | None) -> str:
    """Open $EDITOR for free-text input. Pre-populates with a brief help
    template (or the user's previous description on re-prompt). Returns the
    body with comment lines stripped."""
    template_lines = [
        "# Describe what you want to follow. Be specific:",
        "#   - name people, companies, products you actually read",
        "#   - say what you DON'T want too (e.g. 'no funding rounds')",
        "# Lines starting with # are stripped.",
        "#",
        "# Save and close to continue. Empty file = cancel.",
        "",
    ]
    if initial:
        template_lines.append(initial)
        template_lines.append("")
    edited = _edit_in_editor("\n".join(template_lines))
    return "\n".join(
        line for line in edited.splitlines() if not line.lstrip().startswith("#")
    ).strip()


def _edit_in_editor(content: str) -> str:
    """Drop content into a tempfile, hand it to $EDITOR (or nano/vi), return
    whatever the user saved. Used for both description input and YAML edits."""
    import contextlib

    editor = os.environ.get("EDITOR") or shutil.which("nano") or "vi"
    fd, path = tempfile.mkstemp(suffix=".yaml", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        subprocess.call([editor, path])
        return Path(path).read_text(encoding="utf-8")
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


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


def write_drafted_prefs(yaml_text: str) -> None:
    """Persist an LLM-drafted preferences.yaml. Backs up any existing file
    so the user can recover their hand-written prefs if the draft replaces
    something they cared about."""
    if PREFS_PATH.exists():
        backup = PREFS_PATH.with_suffix(".yaml.bak")
        PREFS_PATH.replace(backup)
        print(f"  (backed up existing preferences.yaml → {backup.name})")
    PREFS_PATH.write_text(yaml_text.rstrip() + "\n", encoding="utf-8")


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
    if state.drafted_yaml is not None:
        print(f"  preferences.yaml  → {PREFS_PATH}  (LLM-drafted)")
    elif state.preset_path is not None:
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
