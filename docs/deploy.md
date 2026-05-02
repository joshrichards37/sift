# Deploying sift

Sift is a single Python process plus an Ollama instance. There's no Docker compose, no K8s, no managed runtime by design — it runs anywhere a Python 3.12 + uv + Ollama install can run.

## On a workstation (the default)

The simplest setup. Two options:

### tmux (least friction)

```bash
tmux new -s sift
uv run sift
# Ctrl+B, D to detach. `tmux a -t sift` to reattach.
```

Doesn't survive reboot. Fine if your workstation rarely reboots.

### systemd user unit (survives reboots, no root needed)

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/sift.service <<'EOF'
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
EOF

systemctl --user daemon-reload
systemctl --user enable --now sift
loginctl enable-linger $USER       # so it survives logout / boot
```

Logs: `journalctl --user -u sift -f`.

### macOS launchd

Create `~/Library/LaunchAgents/sh.sift.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>sh.sift</string>
  <key>WorkingDirectory</key>  <string>/Users/YOU/workspace/sift</string>
  <key>ProgramArguments</key>  <array><string>/Users/YOU/.local/bin/uv</string><string>run</string><string>sift</string></array>
  <key>RunAtLoad</key>         <true/>
  <key>KeepAlive</key>         <true/>
  <key>StandardOutPath</key>   <string>/tmp/sift.log</string>
  <key>StandardErrorPath</key> <string>/tmp/sift.err</string>
</dict>
</plist>
```

Then `launchctl load ~/Library/LaunchAgents/sh.sift.plist`.

## On a small VPS

The bot itself is light (~50MB Python process); the LLM is the expensive part. Three patterns:

### Pattern 1 — VPS for bot, Ollama at home over Tailscale

You keep the GPU on your home machine. The bot runs on a $5-10/mo VPS that's always reachable to Telegram. They talk over a Tailscale tunnel.

- Pro: your hardware does the inference (free, GPU-accelerated).
- Con: when home machine is off, agent can't generate digests.
- VPS: any 1GB-RAM linux box (Hetzner CX11, DigitalOcean basic).
- Tailscale: install on both, set `LLM_BASE_URL=http://<home-tailnet-name>:11434/v1` on the VPS.

### Pattern 2 — VPS only, hosted LLM

No home dependency. Bot + Ollama-compatible API both in the cloud.

- Pro: 24/7 uptime, no home-machine entanglement.
- Con: pay for LLM either via cloud (Gemini Flash free tier handles a small group; Groq same) or rent GPU minutes ($$$).
- Recommended: VPS + Gemini Flash free tier. ~1500 req/day handles one user comfortably.

```bash
# .env on the VPS
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=google/gemini-2.5-flash
```

### Pattern 3 — VPS with a GPU droplet

Lambda, Modal, RunPod, vast.ai sell GPU-by-hour. For a personal news bot you'd pay $30-100/mo for an always-on GPU — roughly the wrong shape for the workload. Skip unless you have other LLM needs.

## On a Raspberry Pi or NUC

The bot fits, the local LLM doesn't. Use Pattern 2 or 3 above.

## What to monitor

The agent logs to stdout (or to journalctl/launchd as configured). Useful tail patterns:

- `source <id>: fetched N items` — every successful poll.
- `source <id>: poll failed` followed by traceback — transient or persistent. The poll loop sleeps and retries.
- `next digest at <iso> (in N min)` — every successful digest cycle prints this on completion.
- `HTML send to <chat_id> failed; retrying as plain text` — should be rare; investigate if frequent.

For real monitoring (uptime alerts, etc.), point your existing system at the process or the systemd unit's status. Sift doesn't expose its own metrics endpoint by design.

## Backups

Two things to back up:

- `preferences.yaml` — your topic config. Throw it in dotfiles.
- `sift.db` (or whatever `DB_PATH` is) — SQLite file with your scored articles, summaries, feedback history, and source cursors.

The DB is rebuilt-from-feeds-on-day-1 if you lose it (you'll just re-score everything you already saw). Worth backing up if you've accumulated useful feedback (👍/👎) over time.

`.env` — keep it on the box, don't commit. Telegram bot tokens are revocable through @BotFather if leaked.

## Restarting

Edit `preferences.yaml`, `.env`, or any `*.py`, then:

- **tmux**: `Ctrl+C` in the pane, `uv run sift`.
- **systemd**: `systemctl --user restart sift`.
- **launchd**: `launchctl kickstart -k gui/$UID/sh.sift`.

There's no hot-reload. Restart is cheap (~2s) so this is fine.
