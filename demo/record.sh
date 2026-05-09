#!/usr/bin/env bash
# Regenerate docs/img/wizard-prefs.gif from demo/wizard-prefs.tape.
#
# Backs up the user's preferences.yaml around the run because the wizard
# writes to it on the Accept path. Without the backup, anyone regenerating
# the GIF would silently clobber their personal prefs.
#
# Requires:
#   - vhs        (sudo pacman -S vhs   /   brew install vhs)
#   - ttyd       (pulled in as a vhs dep on most distros)
#   - A working .env with valid LLM_* — the recording exercises a real
#     LLM call against the configured backend.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v vhs >/dev/null 2>&1; then
    echo "vhs is not installed. See https://github.com/charmbracelet/vhs" >&2
    exit 1
fi

if [[ -f preferences.yaml ]]; then
    cp -f preferences.yaml preferences.yaml.demo-backup
    trap 'mv -f preferences.yaml.demo-backup preferences.yaml' EXIT
fi

vhs demo/wizard-prefs.tape
echo
echo "Generated: docs/img/wizard-prefs.gif"
