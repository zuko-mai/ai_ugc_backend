"""Claims review — flag invented facts not grounded in product inputs."""

from __future__ import annotations

import re

from app.stages.hook_text_engine.schemas import ClaimsReview

# Patterns that typically imply unstated proof / results / testimonials.
_RISKY_CLAIM_PATTERNS = [
    re.compile(r"\b\d+(?:[.,]\d+)?\s*%"),
    re.compile(r"\b\d+(?:[.,]\d+)?\s*[xX]\b"),
    re.compile(r"\b(guaranteed|proven|clinically|scientifically)\b", re.I),
    re.compile(r"\b(reviews?|testimonials?|customers? say|before.?after)\b", re.I),
    re.compile(r"\b(lose|gained?|saved?)\s+\d+", re.I),
    re.compile(r"\b(results? in|#\d+\s*(best|seller))\b", re.I),
]


def review_claims(
    hook_text: str,
    *,
    product_name: str,
    product_description: str,
) -> ClaimsReview:
    """Return pass/fail/review based on grounding in provided product copy."""
    grounded = f"{product_name}\n{product_description}".lower()
    text = hook_text.strip()
    if not text:
        return "fail"

    risky_hits: list[str] = []
    for pattern in _RISKY_CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            snippet = match.group(0)
            # Allow if the exact risky phrase already appears in product inputs.
            if snippet.lower() not in grounded and not _number_grounded(snippet, grounded):
                risky_hits.append(snippet)

    if risky_hits:
        return "fail"

    # Soft check: product name mention improves trust but is not required.
    return "pass"


def product_alignment_score(
    hook_text: str,
    *,
    product_name: str,
    product_description: str,
    target_audience: str,
) -> float:
    """Lexical overlap score in [0, 1] — no invented semantics."""
    hook_tokens = _tokens(hook_text)
    if not hook_tokens:
        return 0.0
    ref_tokens = _tokens(f"{product_name} {product_description} {target_audience}")
    if not ref_tokens:
        return 0.0
    overlap = len(hook_tokens & ref_tokens)
    # Weight product name presence.
    name_bonus = 0.15 if _tokens(product_name) & hook_tokens else 0.0
    base = overlap / max(len(hook_tokens), 1)
    return round(min(1.0, base + name_bonus), 3)


def detect_structure(text: str) -> str:
    t = text.strip()
    if not t:
        return "empty"
    if t.endswith("?"):
        return "question"
    if re.search(r"\b(but|however|vs\.?|versus)\b", t, re.I):
        return "contrast"
    if re.search(r"^\d+[\.\)]", t) or re.search(r"\b\d+\s+(ways|tips|reasons)\b", t, re.I):
        return "listicle"
    if re.search(r"\b(stop|don'?t|never|wait)\b", t, re.I):
        return "pattern_interrupt"
    if len(t.split()) <= 8:
        return "short_punch"
    return "statement"


def _number_grounded(snippet: str, grounded: str) -> bool:
    nums = re.findall(r"\d+(?:\.\d+)?", snippet)
    return bool(nums) and all(n in grounded for n in nums)


def _tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9']+", text.lower())
        if len(tok) > 2
    }
