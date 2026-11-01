from app.stages.hook_text_engine.claims import (
    detect_structure,
    product_alignment_score,
    review_claims,
)


def test_claims_pass_when_grounded_in_description() -> None:
    result = review_claims(
        "GlowSerum helps dry skin feel softer",
        product_name="GlowSerum",
        product_description="GlowSerum helps dry skin feel softer with daily use.",
    )
    assert result == "pass"


def test_claims_fail_on_invented_percent() -> None:
    result = review_claims(
        "GlowSerum gives 97% better skin in 3 days",
        product_name="GlowSerum",
        product_description="A gentle face serum for dry skin.",
    )
    assert result == "fail"


def test_claims_allow_percent_if_in_product_copy() -> None:
    result = review_claims(
        "97% of testers noticed softer skin",
        product_name="GlowSerum",
        product_description="In our study, 97% of testers noticed softer skin.",
    )
    assert result == "pass"


def test_product_alignment_rewards_name_overlap() -> None:
    score = product_alignment_score(
        "Try GlowSerum today",
        product_name="GlowSerum",
        product_description="Face serum",
        target_audience="busy parents",
    )
    assert score > 0


def test_detect_structure_question() -> None:
    assert detect_structure("Have you tried this yet?") == "question"
