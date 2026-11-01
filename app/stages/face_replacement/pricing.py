"""Paid-tier cost estimate for face-replacement model providers.

Stages stay independent — rates live here (not imported from hook_analyzer).
"""

from __future__ import annotations

from dataclasses import dataclass

# USD per 1M tokens: (input_per_1m, output_per_1m)
_MODEL_RATES: dict[str, tuple[float, float]] = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-2.5-flash": (0.30, 2.50),
    # Nano Banana family (approximate paid-tier token rates; image billing may also apply)
    "gemini-3-pro-image": (2.00, 12.00),  # Nano Banana Pro
    "gemini-3.1-flash-image": (0.30, 2.50),  # Nano Banana 2
    "gemini-nano-banana-2.1": (0.30, 2.50),
    "gemini-2.5-flash-image": (0.30, 2.50),
    "gemini-2.0-flash-preview-image-generation": (0.10, 0.40),
}

_DEFAULT_RATES = (2.00, 12.00)  # default stack = Nano Banana Pro


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
    pricing_note: str = ""


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
        "Free tier may be $0; image models may also bill per-image."
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


def empty_usage(*, model: str) -> TokenUsage:
    return estimate_cost(
        model=model,
        prompt_tokens=0,
        candidates_tokens=0,
        thoughts_tokens=0,
        total_tokens=0,
    )
