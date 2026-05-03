from __future__ import annotations

from sift.bench import (
    WORKLOAD_SCORING_PER_DAY,
    WORKLOAD_SUMMARY_PER_DAY,
    BenchResult,
    CallStats,
    format_report,
    percentile,
    summarise,
)


def test_percentile_inclusive_nearest_rank() -> None:
    """Nearest-rank percentile rather than linear interpolation. With 5 samples,
    p50 lands exactly on the middle value; p100 is the max; p0 is the min.
    Interpolation would give finer-grained numbers but at benchmark sample sizes
    that's misleading precision."""
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(vals, 50) == 3.0  # rank = 0.5 * 4 = 2.0 → sorted[2] = 3
    assert percentile(vals, 100) == 5.0
    assert percentile(vals, 0) == 1.0
    # p95 with 10 samples: rank = 0.95 * 9 = 8.55 → round to 9 → sorted[9] = 10
    assert percentile([float(i) for i in range(1, 11)], 95) == 10.0


def test_percentile_handles_edge_cases() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([42.0], 95) == 42.0


def test_summarise_projects_daily_workload() -> None:
    """Standard sift workload: 100 scorings + 10 summaries per user per day.
    Token projections must multiply averaged per-call counts by those constants
    so the cost/day number is grounded in the same assumption everywhere."""
    result = BenchResult(
        backend_url="http://test/v1",
        model="m",
        scoring=[CallStats(latency_ms=100, input_tokens=500, output_tokens=50) for _ in range(10)],
        summary=[CallStats(latency_ms=300, input_tokens=1500, output_tokens=150) for _ in range(5)],
    )
    s = summarise(result, input_cost_per_m=0.59, output_cost_per_m=0.79)

    # 100 × 500 + 10 × 1500 = 65 000 input tokens / user / day
    assert (
        s.projected_daily_input_tokens
        == 500 * WORKLOAD_SCORING_PER_DAY + 1500 * WORKLOAD_SUMMARY_PER_DAY
    )
    assert s.projected_daily_input_tokens == 65_000
    # 100 × 50 + 10 × 150 = 6 500 output tokens / user / day
    assert (
        s.projected_daily_output_tokens
        == 50 * WORKLOAD_SCORING_PER_DAY + 150 * WORKLOAD_SUMMARY_PER_DAY
    )
    assert s.projected_daily_output_tokens == 6_500
    # Cost: 65k × 0.59/1M + 6.5k × 0.79/1M = 0.03835 + 0.005135 ≈ 0.043485
    assert s.projected_daily_cost_usd is not None
    assert abs(s.projected_daily_cost_usd - 0.043485) < 1e-6


def test_summarise_omits_cost_when_pricing_not_provided() -> None:
    """Local backends have no cost — running without --input-cost / --output-cost
    must produce a clean report with cost reported as None, not zero. Zero would
    be misleading: 'free!' instead of 'unknown'."""
    result = BenchResult(
        backend_url="http://localhost:11434/v1",
        model="qwen3:30b",
        scoring=[CallStats(latency_ms=200, input_tokens=400, output_tokens=40)],
        summary=[CallStats(latency_ms=500, input_tokens=1200, output_tokens=120)],
    )
    s = summarise(result, input_cost_per_m=None, output_cost_per_m=None)
    assert s.projected_daily_cost_usd is None


def test_summarise_projects_wall_time_for_full_day() -> None:
    """Wall-time projection is the rate-limit / throughput sanity check —
    if the projection exceeds 24h for one user, the backend is too slow even
    for self-hosting one user, never mind a hosted SaaS."""
    result = BenchResult(
        backend_url="x",
        model="m",
        scoring=[CallStats(latency_ms=100) for _ in range(5)],
        summary=[CallStats(latency_ms=300) for _ in range(5)],
    )
    s = summarise(result, input_cost_per_m=None, output_cost_per_m=None)
    # 100ms × 100 + 300ms × 10 = 10 000 + 3 000 = 13 000 ms = 13s
    assert abs(s.projected_daily_seconds - 13.0) < 0.01


def test_format_report_smoke() -> None:
    """The printed report renders without crashing and includes the headline
    numbers a user looks for. Don't lock down exact text — formatting is allowed
    to evolve."""
    result = BenchResult(
        backend_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        scoring=[CallStats(latency_ms=120, input_tokens=500, output_tokens=50)],
        summary=[CallStats(latency_ms=350, input_tokens=1500, output_tokens=150)],
    )
    s = summarise(result, input_cost_per_m=0.59, output_cost_per_m=0.79)
    report = format_report(s)
    assert "groq.com" in report
    assert "llama-3.3-70b-versatile" in report
    assert "Latency" in report
    assert "Cost / user / day" in report


def test_format_report_shows_cost_placeholder_when_none() -> None:
    """When cost couldn't be projected, the report should say so explicitly
    rather than print '$0.0000' (which a user could misread as 'free')."""
    result = BenchResult(
        backend_url="http://localhost:11434/v1",
        model="qwen3",
        scoring=[CallStats(latency_ms=100)],
        summary=[CallStats(latency_ms=300)],
    )
    s = summarise(result, input_cost_per_m=None, output_cost_per_m=None)
    report = format_report(s)
    assert "--input-cost" in report
