# Prompting

The LLM does two jobs: score relevance (1-10), and summarise things that pass. Both prompts live in `src/sift/llm.py` and are designed to be diffable when tuning.

## Relevance scoring

Each ingested article gets one LLM call. Input: your `topics` block + the article title + the first 2000 chars of the body. Output: strict JSON `{"score": 1-10, "reason": "..."}`.

```python
RELEVANCE_SYSTEM = """You score how relevant an article is to a user's interests on a 1-10 scale.

10 = exactly the user's wheelhouse, would push immediately.
7  = clearly related to their stated topics, useful.
5  = adjacent / occasionally interesting.
1  = unrelated noise.

Reply with strict JSON only: {"score": <int 1-10>, "reason": "<one short sentence>"}.
No markdown, no preamble."""
```

If the model returns something other than valid JSON, the article is given score 0 (filtered out, never seen) and the failure is logged. This is the right call for a personal bot — false negatives are recoverable from the source backlog; false positives are *visible noise* in your Telegram.

### Tuning the threshold

Default is 7/10. The first day or two will tell you whether that's right:

- **Too noisy** (5+ articles per digest you skim past): raise to 8.
- **Too quiet** (fewer than 2-3 per digest): lower to 6.
- **Wrong articles passing**: the threshold is fine; sharpen `topics`. The LLM scores against what you write, so vague topics give vague scores.

### What makes a good `topics` block

The LLM has nothing to filter against except the words you write. So:

- **Be specific.** "Foundation-model releases, agent SDKs, Claude Code features" beats "AI".
- **Name the entities you care about.** People, companies, products, brand names. The model will pattern-match on them.
- **Say what you don't want.** A "Score down: …" paragraph at the end of `topics` is read by the LLM and biases scores down for matching content. More effective than `exclude_keywords` for nuanced cases.

`exclude_keywords` is a hard filter — articles whose title or body contains any keyword (case-insensitive substring) are scored 1 without an LLM call. Use it for things you never want to see (`cryptocurrency`, `astrology`). Don't use it for nuance — the LLM does that better.

## Summarisation

Articles that score ≥ threshold get a second LLM call. Input: title + source + first 8000 chars. Output: a `summary_target_words`-word executive summary.

```python
SUMMARY_SYSTEM = """You write executive summaries for a busy technical reader.

- {target_words} words, ±20%.
- Lead with the single most important takeaway.
- Concrete details over generalities. If the article announces a feature, name it.
- No filler ("In this article…", "The author argues…"). Just the substance.
- Plain prose, no bullets unless the article is itself a list of items."""
```

`summary_target_words` defaults to 80 — fits comfortably in a Telegram digest entry. Crank to 120-150 for research-paper presets where 80 words can't carry the load.

### Why no chain-of-thought, no markdown, no bullets?

- **Chain-of-thought** wastes tokens for a simple summarisation task. Use a non-reasoning model.
- **Markdown** mostly doesn't render in Telegram-as-HTML; we'd just be escaping it. Plain prose is right.
- **Bullets** fragment the summary's narrative — they're worse for skim-reading on a phone, despite feeling more "readable" on a desktop.

## Tuning the prompts

Edit `RELEVANCE_SYSTEM` or `SUMMARY_SYSTEM` in `llm.py`, restart the agent. Articles already scored stay at their old score; only newly ingested ones get the new prompt.

If you want to A/B prompts, do it on the same article corpus:

```python
# in a one-off script
from sift.llm import LLM
from sift.config import Settings, load_preferences
from sift.storage import connect

settings = Settings(); prefs = load_preferences(settings.preferences_path)
llm_a = LLM(...)  # current prompts
llm_b = LLM(...)  # candidate prompts (edit a copy of llm.py)

with connect(settings.db_path) as conn:
    rows = conn.execute("SELECT * FROM articles WHERE relevance_score IS NOT NULL LIMIT 50").fetchall()
for r in rows:
    print(r["title"], "current:", await llm_a.score_relevance(...), "candidate:", await llm_b.score_relevance(...))
```

If a prompt change is load-bearing (i.e., real regressions if it's reverted), say so in the commit message body — future-you reading `git log` is the durable signal that the wording matters.

## Cost optimisation

Two levers:

1. **`max_per_cycle`**: the number of articles scored per source per poll. Defaults to 3. With 15 sources, that's ≤45 LLM calls per "round of polls" across all sources — bounded.
2. **`exclude_keywords`**: short-circuits before any LLM call. Cheapest possible filter.

A typical day at default settings: 100-300 LLM calls total. On local Ollama, free. On Gemini Flash free tier, well under quota.
