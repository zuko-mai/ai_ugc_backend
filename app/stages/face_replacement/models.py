"""Recommended Gemini / Nano Banana image models for face replacement."""

from __future__ import annotations

# Default + secondary option (API model ids).
NANO_BANANA_PRO = "gemini-3-pro-image"
NANO_BANANA_2 = "gemini-3.1-flash-image"

DEFAULT_FACE_REPLACEMENT_MODEL = NANO_BANANA_PRO
SECONDARY_FACE_REPLACEMENT_MODEL = NANO_BANANA_2

# Friendly aliases → API model id
MODEL_ALIASES: dict[str, str] = {
    "nano-banana-pro": NANO_BANANA_PRO,
    "nanobanana-pro": NANO_BANANA_PRO,
    "nano_banana_pro": NANO_BANANA_PRO,
    "gemini-3-pro-image": NANO_BANANA_PRO,
    "nano-banana-2": NANO_BANANA_2,
    "nanobanana-2": NANO_BANANA_2,
    "nano_banana_2": NANO_BANANA_2,
    "gemini-3.1-flash-image": NANO_BANANA_2,
}

RECOMMENDED_MODELS = (
    NANO_BANANA_PRO,  # default
    NANO_BANANA_2,  # secondary
)


def resolve_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    return MODEL_ALIASES.get(key, model_name.strip())
