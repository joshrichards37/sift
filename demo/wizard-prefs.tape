# vhs tape for the LLM-drafted-preferences flow.
#
# Drives `sift-setup --resume prefs` so we skip backend + telegram and
# capture only the new wizard feature: free-text → model draft → pre-flight
# → accept. The scripted EDITOR (demo/scripted-editor.sh) writes a canned
# description so the recording is deterministic across runs.
#
# Regenerate after wizard UX changes:
#     bash demo/record.sh

Output docs/img/wizard-prefs.gif

Set FontSize 14
Set Width  1300
Set Height 780
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 60ms
Set Framerate 24

# Wire the scripted editor so the description prompt doesn't block.
Env EDITOR "/home/josh/workspace/sift/demo/scripted-editor.sh"

Type "uv run sift-setup --resume prefs"
Sleep 400ms
Enter

# Header + "Configure preferences now?" — accept default (Yes).
Sleep 2500ms
Enter

# Mode select — default is "Describe what you follow…" (the draft path).
Sleep 1200ms
Enter

# Show the "Asking model…" line, then hide the GIF clock while the model
# thinks AND the pre-flight runs (both happen serially before any user
# input is needed). qwen3:30b can take 20-40s; pre-flight adds 2-4s.
# Budget 60s with Hide so the viewer doesn't see dead air.
Sleep 1500ms
Hide
Sleep 60s
Show

# At this point the screen shows: rendered YAML + pre-flight results +
# auto-disable Y/n prompt (if any sources were flagged).
Sleep 4s
Enter

# Wizard transitions to "What now?" select — Accept and save is default.
Sleep 3s
Enter

# Final summary screen.
Sleep 3s
