"""Hook text adaptation — preserve psychological mechanic, ground in product facts."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from uuid import UUID

from app.core.logging import get_logger
from app.stages.hook_text_engine.claims import (
    detect_structure,
    product_alignment_score,
    review_claims,
)
from app.stages.hook_text_engine.exceptions import AdaptationError
from app.stages.hook_text_engine.schemas import HookGoal, HookVariant

logger = get_logger(__name__)

_ADAPT_SYSTEM = """You adapt UGC video hook texts for a new product.

Rules:
1. Preserve the psychological mechanic of the original (curiosity gap, urgency, pain, benefit, question, contrast, etc.).
2. Keep similar length and tone/style.
3. Adapt wording to the product name, product description, and target audience.
4. NEVER invent product features, results, percentages, reviews, testimonials, or proof not present in the product description.
5. Output JSON only:
{"variants":[{"hook_text":"...","hook_type":"curiosity","structure":"question"}]}
hook_type must be one of: curiosity, urgency, pain, benefit, social_proof, question, contrast.
structure is a short label for the sentence pattern (question, contrast, short_punch, statement, pattern_interrupt, listicle).
"""


async def generate_variants(
    *,
    original_text: str,
    product_name: str,
    product_description: str,
    target_audience: str,
    language: str,
    hook_goal: HookGoal,
    variants_count: int,
    api_key: str,
    model: str,
) -> list[HookVariant]:
    if api_key:
        try:
            raw = await asyncio.to_thread(
                _gemini_adapt,
                original_text,
                product_name,
                product_description,
                target_audience,
                language,
                hook_goal,
                variants_count,
                api_key,
                model,
            )
            variants = [
                _finalize_variant(
                    text=item.get("hook_text", ""),
                    hook_type=item.get("hook_type") or hook_goal,
                    structure=item.get("structure") or detect_structure(str(item.get("hook_text", ""))),
                    product_name=product_name,
                    product_description=product_description,
                    target_audience=target_audience,
                )
                for item in raw
                if str(item.get("hook_text") or "").strip()
            ]
            if variants:
                return variants[:variants_count]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini adaptation failed, using local templates: %s", exc)

    return _local_adapt(
        original_text=original_text,
        product_name=product_name,
        product_description=product_description,
        target_audience=target_audience,
        language=language,
        hook_goal=hook_goal,
        variants_count=variants_count,
    )


def _gemini_adapt(
    original_text: str,
    product_name: str,
    product_description: str,
    target_audience: str,
    language: str,
    hook_goal: HookGoal,
    variants_count: int,
    api_key: str,
    model: str,
) -> list[dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    user = (
        f"Original hook:\n{original_text}\n\n"
        f"Product name: {product_name}\n"
        f"Product description: {product_description}\n"
        f"Target audience: {target_audience}\n"
        f"Language: {language}\n"
        f"Preferred hook_goal: {hook_goal}\n"
        f"Generate exactly {variants_count} variants."
    )
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user)])],
        config=types.GenerateContentConfig(
            system_instruction=_ADAPT_SYSTEM,
            temperature=0.7,
            response_mime_type="application/json",
        ),
    )
    payload = _parse_json(response.text or "")
    variants = payload.get("variants")
    if not isinstance(variants, list) or not variants:
        raise AdaptationError(
            "Gemini returned no adaptation variants",
            details={"model": model},
        )
    return variants


def _local_adapt(
    *,
    original_text: str,
    product_name: str,
    product_description: str,
    target_audience: str,
    language: str,
    hook_goal: HookGoal,
    variants_count: int,
) -> list[HookVariant]:
    """Deterministic, claim-safe templates for offline/tests."""
    _ = language
    structure = detect_structure(original_text)
    benefit = _first_sentence(product_description) or product_name
    audience = target_audience.strip()

    templates_by_goal: dict[str, list[str]] = {
        "curiosity": [
            f"What {audience} notice first about {product_name}?",
            f"Nobody told {audience} this about {product_name}.",
            f"The part about {product_name} that changes everything.",
            f"Why {audience} keep talking about {product_name}.",
            f"{product_name} — but not for the reason you think.",
        ],
        "urgency": [
            f"{audience}: look at {product_name} before it slips by.",
            f"If you care about this, check {product_name} now.",
            f"Don't scroll past {product_name}.",
            f"{product_name} is for {audience} who act today.",
            f"Last chance vibe: {product_name} for {audience}.",
        ],
        "pain": [
            f"Tired of the same problem? {product_name} is built for {audience}.",
            f"{audience} dealing with this — {product_name} speaks to it.",
            f"That frustration? {product_name} was made for it.",
            f"Stop wrestling with it alone — {product_name}.",
            f"For {audience} who are done with the old way: {product_name}.",
        ],
        "benefit": [
            f"{product_name}: {benefit}",
            f"{audience} get this from {product_name}: {benefit}",
            f"Meet {product_name} — {benefit}",
            f"{product_name} helps {audience}. {benefit}",
            f"Built for {audience}: {product_name}.",
        ],
        "social_proof": [
            f"{audience} are paying attention to {product_name}.",
            f"Why {audience} keep recommending {product_name}.",
            f"{product_name} is showing up in {audience} conversations.",
            f"People like {audience} keep choosing {product_name}.",
            f"{product_name} — noticed by {audience}.",
        ],
        "question": [
            f"Have you tried {product_name} yet?",
            f"What if {product_name} is what {audience} needed?",
            f"Still searching? Heard of {product_name}?",
            f"Is {product_name} right for {audience}?",
            f"Ready for {product_name}?",
        ],
        "contrast": [
            f"Not another generic option — {product_name} for {audience}.",
            f"Old way vs {product_name}.",
            f"Skip the noise. {product_name}.",
            f"{product_name} isn't more of the same.",
            f"For {audience}: different approach — {product_name}.",
        ],
    }

    templates = templates_by_goal.get(hook_goal, templates_by_goal["curiosity"])
    # Prefer keeping original length feel: trim/expand lightly.
    target_len = max(4, len(original_text.split()))
    variants: list[HookVariant] = []
    for i in range(variants_count):
        text = templates[i % len(templates)]
        words = text.split()
        if len(words) > target_len + 8:
            text = " ".join(words[: target_len + 6]).rstrip(".,;") + ("?" if original_text.strip().endswith("?") else ".")
        variants.append(
            _finalize_variant(
                text=text,
                hook_type=hook_goal,
                structure=structure,
                product_name=product_name,
                product_description=product_description,
                target_audience=target_audience,
            )
        )
    return variants


def _finalize_variant(
    *,
    text: str,
    hook_type: str,
    structure: str,
    product_name: str,
    product_description: str,
    target_audience: str,
    variant_id: UUID | None = None,
) -> HookVariant:
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    goal: HookGoal = hook_type if hook_type in {
        "curiosity",
        "urgency",
        "pain",
        "benefit",
        "social_proof",
        "question",
        "contrast",
    } else "curiosity"
    return HookVariant(
        variant_id=variant_id or uuid.uuid4(),
        hook_text=cleaned,
        hook_type=goal,
        structure=structure or detect_structure(cleaned),
        product_alignment=product_alignment_score(
            cleaned,
            product_name=product_name,
            product_description=product_description,
            target_audience=target_audience,
        ),
        claims_review=review_claims(
            cleaned,
            product_name=product_name,
            product_description=product_description,
        ),
    )


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return (parts[0] if parts else text).strip()


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"variants": []}
        data = json.loads(match.group(0))
    if isinstance(data, list):
        return {"variants": data}
    return data if isinstance(data, dict) else {"variants": []}
