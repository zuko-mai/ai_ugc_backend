"""Paid-tier estimate for Gemini models used by Hook Analyzer.

Rates from Google AI Studio pricing (intro for gemini-3.8-flash through 2026-12-31).
Free tier is $0 in practice; we still report the paid-tier estimate so costs are visible.
"""

from __future__ import annotations

from dataclasses import dataclass

# USD per 1M tokens
_MODEL_RATES: dict[str, tuple[float, float]] = {
    # model: (input_per_1m, output_per_1m) — output includes thinking tokens
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}

_DEFAULT_RATES = (0.25, 1.50)


@dataclass(frozen=True)
class TokenUsage:
    model: str
    prompt_tokens: int
    candidates_tokens: int
    thoughts_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    estimated_cost_usd: float
    currency: str = "USD"
    pricing_note: str = (
        "Paid-tier estimate for gemini-3.1-flash-lite "
        "($0.25 / $1.50 per 1M input/output). Free tier may be $0."
    )


def estimate_cost(
    *,
    model: str,
    prompt_tokens: int,
    candidates_tokens: int,
    thoughts_tokens: int = 0,
    total_tokens: int | None = None,
) -> TokenUsage:
    input_rate, output_rate = _MODEL_RATES.get(model, _DEFAULT_RATES)
    output_tokens = max(0, candidates_tokens) + max(0, thoughts_tokens)
    input_cost = (max(0, prompt_tokens) / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    total = total_tokens
    if total is None:
        total = max(0, prompt_tokens) + output_tokens

    note = (
        f"Paid-tier estimate for {model} "
        f"(${input_rate:.2f} / ${output_rate:.2f} per 1M input/output). "
        "Free tier may be $0."
    )
    return TokenUsage(
        model=model,
        prompt_tokens=max(0, prompt_tokens),
        candidates_tokens=max(0, candidates_tokens),
        thoughts_tokens=max(0, thoughts_tokens),
        total_tokens=max(0, total),
        input_cost_usd=round(input_cost, 8),
        output_cost_usd=round(output_cost, 8),
        estimated_cost_usd=round(input_cost + output_cost, 8),
        pricing_note=note,
    )
