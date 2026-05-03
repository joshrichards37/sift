"""Backend benchmark tool.

Runs a fixed set of synthetic articles through the configured LLM backend's
scoring + summary paths, captures per-call latency and token usage, and
projects per-user/day cost given the standard sift workload (100 articles
scored to deliver 10).

Use this to validate any backend (Ollama, llama.cpp, MLX-LM, Groq, OpenRouter,
OpenAI, etc.) before committing to it for a 24h run.

Usage:
    LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=... \\
        uv run sift-bench [options]

Options (with --help for the full list):
    --n-scoring N        scoring calls to run (default 30)
    --n-summary N        summary calls to run (default 10)
    --input-cost FLOAT   $/1M input tokens for cost projection
    --output-cost FLOAT  $/1M output tokens
    --json               machine-readable output
"""

from __future__ import annotations

import argparse
import asyncio
import json as jsonlib
import statistics
import sys
import time
from dataclasses import dataclass, field

from sift.config import Preferences, Settings, SourcePref
from sift.llm import LLM
from sift.sources.base import Article

# Synthetic articles spanning common digest content shapes (release announcement,
# security advisory, technical research, opinion). Body lengths are representative
# of real RSS/HN/Reddit payloads so token counts project realistically.
SCORING_ARTICLES: list[Article] = [
    Article(
        source_id="hn",
        url="https://example.com/openai-release",
        title="OpenAI announces GPT-5 with improved reasoning and tool use",
        body=(
            "OpenAI today released GPT-5, claiming a 30% improvement in coding "
            "benchmarks (HumanEval pass@1 from 87% to 91%) and a meaningful drop "
            "in hallucination rate compared to GPT-4o. The model is available "
            "immediately via the API at $5/M input tokens and $15/M output tokens. "
            "Tool use latency is reportedly ~2x faster, and structured outputs now "
            "guarantee schema conformance without retries. A new 'thinking' mode "
            "is gated behind a separate model id and pricing tier."
        ),
    ),
    Article(
        source_id="rss:ars-technica",
        url="https://example.com/security-zero-day",
        title="Cloud provider discloses zero-day in container runtime",
        body=(
            "AWS disclosed Tuesday that a previously unreported vulnerability in "
            "containerd allowed cross-container escapes under specific kernel "
            "configurations. Customers using Fargate are unaffected; ECS users on "
            "older AMIs should patch immediately. The CVE was reported by an "
            "internal red team and there is no evidence of exploitation in the wild. "
            "A coordinated disclosure window of 90 days expired today."
        ),
    ),
    Article(
        source_id="reddit:LocalLLaMA",
        url="https://example.com/quant-comparison",
        title="Q4_K_M vs Q5_K_S on Qwen3-30B for code completion",
        body=(
            "After running HumanEval and MBPP on both quantizations of "
            "Qwen3-30B-A3B-Instruct, Q5_K_S shows a 1.2% pass-rate improvement "
            "over Q4_K_M but uses 18% more VRAM (3.2GB vs 2.7GB active experts). "
            "For most use cases the Q4 variant is the better tradeoff — the quality "
            "delta is within noise on most real-world tasks."
        ),
    ),
    Article(
        source_id="rss:simonwillison",
        url="https://example.com/agent-pattern",
        title="A pattern for agent loops that don't get stuck",
        body=(
            "Most agent loops I've seen fail in two ways: they enter degenerate "
            "self-correction spirals on ambiguous tasks, or they declare success "
            "without actually verifying the goal state. A simple guard helps both: "
            "after every N tool calls, force the agent to summarise progress against "
            "the original prompt and explicitly mark the goal as 'achieved' / "
            "'blocked' / 'in-progress'. Bonus: this also makes the loop interruptible."
        ),
    ),
    Article(
        source_id="hn",
        url="https://example.com/postgres-perf",
        title="Why your Postgres queries are slow when joining on JSONB",
        body=(
            "JSONB columns don't get hash-join treatment by default — Postgres "
            "estimates extreme cardinality and falls back to nested loops. The fix "
            "is usually a partial expression index on the specific JSONB path you're "
            "joining on, or extracting the key into a generated column with btree. "
            "We saw a 200x speedup on a 50M-row join after this change."
        ),
    ),
]

# Summarisation uses longer payloads (the full article body, not the snippet),
# so distinct fixtures with more text. Token counts will be higher per call.
SUMMARY_ARTICLES: list[Article] = [
    Article(
        source_id=a.source_id,
        url=a.url,
        title=a.title,
        body=a.body * 3,  # ~3x lengthier — closer to typical full-article body
    )
    for a in SCORING_ARTICLES
]


SYNTHETIC_PREFS_TOPICS = (
    "- Foundation model releases — Claude, GPT, Gemini, open-source weights — when "
    "they meaningfully change capability rather than incremental quality bumps.\n"
    "- Security disclosures with architectural implications — CVEs that change how "
    "services should be designed, not vendor advisories.\n"
    "- Local LLM optimization, quantization tradeoffs, inference engines.\n"
    "- Agent harness engineering, tool-use patterns, evaluation methodologies.\n"
    "- Database performance and query planning insights."
)


# Standard workload assumed for cost projection. Calibrated against typical sift
# usage: a user receives ~10 articles per digest, threshold filters out ~90% of
# scored articles, so to deliver 10 we score ~100. Summaries only run on the
# 10 that pass threshold.
WORKLOAD_SCORING_PER_DAY = 100
WORKLOAD_SUMMARY_PER_DAY = 10


@dataclass
class CallStats:
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class BenchResult:
    backend_url: str
    model: str
    scoring: list[CallStats] = field(default_factory=list)
    summary: list[CallStats] = field(default_factory=list)


async def run_benchmark(
    llm: LLM, *, n_scoring: int, n_summary: int, prefs: Preferences | None = None
) -> BenchResult:
    """Run n_scoring + n_summary calls against the LLM. Sequential — measures
    single-stream latency, not throughput under concurrency. Sift's production
    scheduler is also sequential per source, so this matches reality."""
    prefs = prefs or _synthetic_prefs()
    result = BenchResult(backend_url=str(llm.client.base_url), model=llm.model)

    for i in range(n_scoring):
        article = SCORING_ARTICLES[i % len(SCORING_ARTICLES)]
        t0 = time.perf_counter()
        await llm.score_relevance(article, prefs)
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = llm.last_usage or (0, 0)
        result.scoring.append(
            CallStats(latency_ms=latency_ms, input_tokens=usage[0], output_tokens=usage[1])
        )

    for i in range(n_summary):
        article = SUMMARY_ARTICLES[i % len(SUMMARY_ARTICLES)]
        t0 = time.perf_counter()
        await llm.summarize(article, prefs)
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = llm.last_usage or (0, 0)
        result.summary.append(
            CallStats(latency_ms=latency_ms, input_tokens=usage[0], output_tokens=usage[1])
        )

    return result


def _synthetic_prefs() -> Preferences:
    return Preferences(
        topics=SYNTHETIC_PREFS_TOPICS,
        sources=[SourcePref(id="hn", query="x")],
        summary_target_words=50,
    )


def percentile(values: list[float], p: float) -> float:
    """Inclusive nearest-rank percentile. p in [0, 100]. Empty list → 0."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    return sorted_vals[round(rank)]


@dataclass
class Summary:
    """Aggregated bench numbers, ready to print or serialise."""

    backend_url: str
    model: str
    n_scoring: int
    n_summary: int
    scoring_p50_ms: float
    scoring_p95_ms: float
    summary_p50_ms: float
    summary_p95_ms: float
    avg_scoring_input_tokens: float
    avg_scoring_output_tokens: float
    avg_summary_input_tokens: float
    avg_summary_output_tokens: float
    projected_daily_input_tokens: int
    projected_daily_output_tokens: int
    projected_daily_cost_usd: float | None  # None if costs not provided
    projected_daily_seconds: float


def summarise(
    result: BenchResult, *, input_cost_per_m: float | None, output_cost_per_m: float | None
) -> Summary:
    """Reduce a BenchResult to headline numbers + workload projection."""
    s_lat = [c.latency_ms for c in result.scoring]
    m_lat = [c.latency_ms for c in result.summary]
    avg_s_in = statistics.fmean(c.input_tokens for c in result.scoring) if result.scoring else 0.0
    avg_s_out = statistics.fmean(c.output_tokens for c in result.scoring) if result.scoring else 0.0
    avg_m_in = statistics.fmean(c.input_tokens for c in result.summary) if result.summary else 0.0
    avg_m_out = statistics.fmean(c.output_tokens for c in result.summary) if result.summary else 0.0

    daily_in = int(avg_s_in * WORKLOAD_SCORING_PER_DAY + avg_m_in * WORKLOAD_SUMMARY_PER_DAY)
    daily_out = int(avg_s_out * WORKLOAD_SCORING_PER_DAY + avg_m_out * WORKLOAD_SUMMARY_PER_DAY)
    cost: float | None = None
    if input_cost_per_m is not None and output_cost_per_m is not None:
        cost = (daily_in / 1_000_000) * input_cost_per_m + (
            daily_out / 1_000_000
        ) * output_cost_per_m

    avg_s_lat = statistics.fmean(s_lat) if s_lat else 0.0
    avg_m_lat = statistics.fmean(m_lat) if m_lat else 0.0
    daily_seconds = (
        avg_s_lat * WORKLOAD_SCORING_PER_DAY + avg_m_lat * WORKLOAD_SUMMARY_PER_DAY
    ) / 1000.0

    return Summary(
        backend_url=result.backend_url,
        model=result.model,
        n_scoring=len(result.scoring),
        n_summary=len(result.summary),
        scoring_p50_ms=percentile(s_lat, 50),
        scoring_p95_ms=percentile(s_lat, 95),
        summary_p50_ms=percentile(m_lat, 50),
        summary_p95_ms=percentile(m_lat, 95),
        avg_scoring_input_tokens=avg_s_in,
        avg_scoring_output_tokens=avg_s_out,
        avg_summary_input_tokens=avg_m_in,
        avg_summary_output_tokens=avg_m_out,
        projected_daily_input_tokens=daily_in,
        projected_daily_output_tokens=daily_out,
        projected_daily_cost_usd=cost,
        projected_daily_seconds=daily_seconds,
    )


def format_report(s: Summary) -> str:
    cost_line = (
        f"  Cost / user / day:        ${s.projected_daily_cost_usd:.4f}"
        if s.projected_daily_cost_usd is not None
        else "  Cost / user / day:        (pass --input-cost / --output-cost to project)"
    )
    score_io = f"{s.avg_scoring_input_tokens:>8.0f}    / {s.avg_scoring_output_tokens:>8.0f}"
    summ_io = f"{s.avg_summary_input_tokens:>8.0f}    / {s.avg_summary_output_tokens:>8.0f}"
    workload = f"({WORKLOAD_SCORING_PER_DAY} scored, {WORKLOAD_SUMMARY_PER_DAY} summarised)"
    daily_tok = f"{s.projected_daily_input_tokens:>10,} / {s.projected_daily_output_tokens:>10,}"
    return f"""sift-bench — backend validation
================================
Backend:                    {s.backend_url}
Model:                      {s.model}
Calls:                      {s.n_scoring} scoring, {s.n_summary} summary

Latency
-------
  Scoring  p50 / p95:       {s.scoring_p50_ms:>8.0f} ms / {s.scoring_p95_ms:>8.0f} ms
  Summary  p50 / p95:       {s.summary_p50_ms:>8.0f} ms / {s.summary_p95_ms:>8.0f} ms

Token usage (per call, average)
--------------------------------
  Scoring  in / out:        {score_io}
  Summary  in / out:        {summ_io}

Projection — 1 user / day  {workload}
-----------------------------------------------------------------
  Tokens in / out:          {daily_tok}
  Wall time:                {s.projected_daily_seconds:>8.1f} s
{cost_line}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sift-bench",
        description="Benchmark the configured LLM backend against synthetic articles.",
    )
    parser.add_argument(
        "--n-scoring", type=int, default=30, help="number of scoring calls (default 30)"
    )
    parser.add_argument(
        "--n-summary", type=int, default=10, help="number of summary calls (default 10)"
    )
    parser.add_argument(
        "--input-cost",
        type=float,
        default=None,
        help="$/1M input tokens for cost projection (e.g. 0.59 Groq Llama 3.3, "
        "0.15 OpenAI gpt-4o-mini)",
    )
    parser.add_argument(
        "--output-cost",
        type=float,
        default=None,
        help="$/1M output tokens (e.g. 0.79 Groq Llama 3.3, 0.60 OpenAI gpt-4o-mini)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    settings = Settings()
    llm = LLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
    result = asyncio.run(run_benchmark(llm, n_scoring=args.n_scoring, n_summary=args.n_summary))
    summary = summarise(
        result, input_cost_per_m=args.input_cost, output_cost_per_m=args.output_cost
    )
    if args.json:
        print(jsonlib.dumps(summary.__dict__, indent=2))
    else:
        print(format_report(summary))
    return 0


def run() -> None:
    sys.exit(main())
