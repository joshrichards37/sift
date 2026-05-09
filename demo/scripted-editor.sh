#!/usr/bin/env bash
# Stand-in $EDITOR for the vhs recording. The wizard's _edit_in_editor()
# launches $EDITOR with a tempfile path; this script writes a canned
# description to that file and exits — bypassing real interactive input
# so the recording stays deterministic.
#
# Used only by demo/wizard-prefs.tape.
set -euo pipefail

cat > "$1" <<'EOF'
I want to track new releases for vLLM, llama.cpp, and Ollama on GitHub,
plus Simon Willison's blog and the LocalLLaMA subreddit.

Score down: funding rounds, generic AI thinkpieces.
EOF
