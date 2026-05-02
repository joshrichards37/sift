# Agent Instructions

For project architecture, conventions, and gotchas, see [`CLAUDE.md`](CLAUDE.md). What follows is generic agent-environment hygiene that applies regardless of the project.

## Non-Interactive Shell Commands

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input. Always use non-interactive flags:

```bash
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

Other commands that may prompt:

- `scp` / `ssh` — `-o BatchMode=yes`
- `apt-get` — `-y`
- `brew` — `HOMEBREW_NO_AUTO_UPDATE=1`
- `git rebase`, `git add` — never use the `-i` flag (interactive mode is unsupported)
