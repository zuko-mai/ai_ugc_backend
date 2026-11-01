from app.stages.face_replacement.pricing import empty_usage, estimate_cost


def test_estimate_cost_image_model() -> None:
    usage = estimate_cost(
        model="gemini-3-pro-image",
        prompt_tokens=2_000,
        candidates_tokens=500,
        thoughts_tokens=100,
    )
    assert usage.total_tokens == 2_600
    assert usage.input_cost_usd > 0
    assert usage.output_cost_usd > 0
    assert usage.estimated_cost_usd == round(
        usage.input_cost_usd + usage.output_cost_usd, 8
    )
    assert usage.currency == "USD"
    assert "gemini-3-pro-image" in usage.pricing_note


def test_resolve_nano_banana_aliases() -> None:
    from app.stages.face_replacement.models import (
        NANO_BANANA_2,
        NANO_BANANA_PRO,
        resolve_model_name,
    )

    assert resolve_model_name("nano-banana-pro") == NANO_BANANA_PRO
    assert resolve_model_name("nano-banana-2") == NANO_BANANA_2
    assert resolve_model_name("gemini-3.1-flash-image") == NANO_BANANA_2


def test_empty_usage() -> None:
    usage = empty_usage(model="stub-passthrough")
    assert usage.total_tokens == 0
    assert usage.estimated_cost_usd == 0.0
