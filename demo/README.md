# Demo recordings

Source files for the GIFs embedded in the project README.

## Files

- `wizard-prefs.tape` — vhs script driving `sift-setup --resume prefs` to
  showcase the LLM-drafted preferences flow (free-text → model draft →
  pre-flight check → accept).
- `scripted-editor.sh` — `$EDITOR` stand-in. The wizard's description prompt
  opens an editor on a tempfile; this script writes a canned description
  and exits so the recording is deterministic. Only used by the tape file.
- `record.sh` — regenerates `docs/img/wizard-prefs.gif`. Backs up
  `preferences.yaml` around the run (the wizard's Accept path writes to it).

## Regenerate

```bash
# Install vhs + ttyd if needed (one-time)
sudo pacman -S vhs       # Arch — pulls in ttyd
# or
brew install vhs         # macOS

# Make sure your .env has working LLM_* — the recording calls a real LLM
bash demo/record.sh
```

Output goes to `docs/img/wizard-prefs.gif`. Commit both the `.tape` and the
GIF so reviewers can see the diff intent.

## Re-recording after wizard changes

If the wizard's prefs-stage UX changes (new prompts, different question
wording, extra confirm step), the timing in `wizard-prefs.tape` will drift.
Symptoms: keys typed during a still-loading prompt, or the recording ending
mid-summary.

Adjust the `Sleep` values in the tape file in proportion to whatever
shifted, or add `Hide` / `Show` blocks around slow sections to keep the
final GIF tight.
