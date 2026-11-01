import pytest

from app.stages.hook_text_engine.adaptation import generate_variants


@pytest.mark.asyncio
async def test_local_adaptation_preserves_count_and_grounds_product() -> None:
    variants = await generate_variants(
        original_text="Wait… nobody talks about this?",
        product_name="AquaBottle",
        product_description="Insulated bottle that keeps drinks cold for hours.",
        target_audience="commuters",
        language="en",
        hook_goal="curiosity",
        variants_count=3,
        api_key="",  # force local templates
        model="unused",
    )
    assert len(variants) == 3
    for variant in variants:
        assert variant.hook_text
        assert variant.hook_type == "curiosity"
        assert variant.claims_review in {"pass", "fail", "review"}
        assert 0 <= variant.product_alignment <= 1
        assert "AquaBottle" in variant.hook_text


@pytest.mark.asyncio
async def test_local_adaptation_does_not_invent_percent_claims() -> None:
    variants = await generate_variants(
        original_text="This changed everything for me",
        product_name="FocusTea",
        product_description="Herbal tea for calm focus.",
        target_audience="students",
        language="en",
        hook_goal="benefit",
        variants_count=5,
        api_key="",
        model="unused",
    )
    for variant in variants:
        assert "%" not in variant.hook_text
        assert variant.claims_review == "pass"
