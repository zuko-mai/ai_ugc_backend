from app.stages.hook_text_engine.combined import _materialize


def test_materialize_builds_variants_and_prefers_on_screen() -> None:
    result = _materialize(
        {
            "on_screen_text": [
                {
                    "text": "Wait for it",
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "confidence": 0.9,
                    "language": "en",
                }
            ],
            "spoken_text": [
                {
                    "text": "you need this",
                    "start_seconds": 0.2,
                    "end_seconds": 1.2,
                    "confidence": 0.8,
                    "language": "en",
                }
            ],
            "preferred_source": "on_screen",
            "original_text": "Wait for it",
            "variants": [
                {
                    "hook_text": "Wait for Saya’s glow drop",
                    "hook_type": "curiosity",
                    "structure": "pattern_interrupt",
                }
            ],
        },
        product_name="Saya",
        product_description="Skincare serum",
        target_audience="Gen Z",
        model="gemini-3.1-flash-lite",
        clip_start=1.5,
        estimated_cost_usd=0.001,
    )
    assert result.preferred_source == "on_screen"
    assert result.original_text == "Wait for it"
    assert len(result.on_screen_text) == 1
    assert len(result.spoken_text) == 1
    assert len(result.variants) == 1
    assert "Saya" in result.variants[0].hook_text
    assert result.model == "gemini-3.1-flash-lite"
    assert result.estimated_cost_usd == 0.001


def test_materialize_falls_back_to_spoken_original() -> None:
    result = _materialize(
        {
            "on_screen_text": [],
            "spoken_text": [
                {
                    "text": "try this serum",
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "language": "en",
                }
            ],
            "preferred_source": "spoken",
            "original_text": "",
            "variants": [{"hook_text": "Try Saya serum tonight", "hook_type": "benefit"}],
        },
        product_name="Saya",
        product_description="Serum",
        target_audience="women",
        model="gemini-2.5-flash",
        clip_start=0.0,
        estimated_cost_usd=None,
    )
    assert result.original_text == "try this serum"
    assert result.preferred_source == "spoken"
