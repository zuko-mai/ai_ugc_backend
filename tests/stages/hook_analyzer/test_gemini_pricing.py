from app.stages.hook_analyzer.gemini_pricing import estimate_cost


def test_estimate_cost_gemini_38_flash() -> None:
    usage = estimate_cost(
        model="gemini-3.8-flash",
        prompt_tokens=1_000_000,
        candidates_tokens=1_000_000,
        thoughts_tokens=0,
    )
    assert usage.input_cost_usd == 0.75
    assert usage.output_cost_usd == 3.75
    assert usage.estimated_cost_usd == 4.5
    assert usage.total_tokens == 2_000_000


def test_thoughts_count_as_output() -> None:
    usage = estimate_cost(
        model="gemini-3.8-flash",
        prompt_tokens=0,
        candidates_tokens=0,
        thoughts_tokens=1_000_000,
    )
    assert usage.output_cost_usd == 3.75
