"""Character-replacement prompt for first-frame generation.

This is full person/identity replacement (not a naive face crop-swap):
IMAGE 1 / FIRST = selected frame to edit (master scene);
IMAGE 2 / SECOND = face reference (identity only).
"""

from __future__ import annotations

CHARACTER_REPLACE_SYSTEM = (
    "You are a photorealistic character-replacement editor for UGC first frames. "
    "Follow the user instructions exactly. "
    "IMAGE 1 (FIRST) is the photo to edit. IMAGE 2 (SECOND) is the identity reference only. "
    "Output only the final edited image."
)

_IMAGE_ROLES = """
IMAGE ROLES (read carefully before editing)

IMAGE 1 = FIRST image = selected_frame = the photo you MUST edit / transform.
It is the master scene. Preserve its pose, body, clothing, background, camera, lighting and composition.

IMAGE 2 = SECOND image = face_reference = identity reference ONLY.
Use it only to know who the person should look like (face / appearance).
Do NOT treat IMAGE 2 as the scene to edit. Do NOT copy its pose, outfit, framing, crop or scale.
""".strip()

_TEXT_REMOVAL_SECTION = """
TEXT REMOVAL

Remove all added text, captions, subtitles, stickers, emojis, watermarks, UI elements and graphic overlays from the FIRST image (IMAGE 1 — the photo being edited).

Do not recreate or replace the removed text.

Naturally reconstruct the areas underneath the removed text so that there are no blurred letters, ghost text, artifacts or empty patches.

The final image should look as if the added text and graphics were never present.

Only preserve text that is physically part of the real-world environment, such as permanent signage, packaging or objects.
""".strip()

_TEXT_KEEP_SECTION = """
TEXT

Leave any existing overlay text, captions, subtitles, stickers, watermarks, UI elements and graphic overlays from the FIRST image (IMAGE 1 — the photo being edited) untouched.

Do not add any new text or graphics.
""".strip()

_CORE_PROMPT = """
Use the FIRST image (IMAGE 1 — photo to edit) as the master scene and the SECOND image (IMAGE 2 — face / identity reference) only as the identity reference.

Replace the person in the FIRST image (IMAGE 1) with the person from the SECOND image (IMAGE 2), while preserving the FIRST image as closely as possible.

The final image must look like the person from the SECOND image was naturally photographed in the exact situation shown in the FIRST image.

IMPORTANT — preserve from the FIRST image (IMAGE 1 — photo to edit):

exact body pose and posture

exact position of the arms, hands, shoulders, torso, hips and legs

exact head position, head tilt and rotation

exact facial expression and emotion

exact eye direction and gaze

exact mouth position and subtle facial tension

exact clothing and outfit

exact clothing colors, fabric, folds, wrinkles and details

exact background and environment

exact objects and furniture

exact camera angle and perspective

exact framing and composition

exact lighting, shadows and color temperature

exact depth of field and photographic characteristics

Use the SECOND image (IMAGE 2 — face / identity reference) ONLY to determine who the person is.

Do NOT copy the pose, clothing, framing, camera distance or apparent scale of the person from the SECOND image.

HEAD AND BODY PROPORTIONS — VERY IMPORTANT

The SECOND image (IMAGE 2 — reference) may be a close-up portrait where the person's face appears much larger than it would naturally appear in the FIRST image (IMAGE 1 — photo to edit).

Do NOT preserve the apparent head size from the SECOND image.

Calculate the person's natural head size relative to the body and perspective of the FIRST image.

The head must have a realistic human head-to-body ratio and must be naturally scaled according to the person's body, camera distance, viewing angle and perspective in the FIRST image.

The head should generally be slightly smaller rather than oversized when necessary to match the proportions of the original person in the FIRST image.

Do not create a large head, oversized face, large eyes, or a face that appears pasted onto the body.

The identity must remain recognizable while the head and face are naturally integrated into the body and perspective of the FIRST image.

CLOTHING

Keep the EXACT clothing from the FIRST image (IMAGE 1 — photo to edit).

Do not transfer any clothing from the SECOND image (IMAGE 2 — reference).

The person from the SECOND image must appear to be wearing the exact outfit from the FIRST image, with realistic fabric behavior, folds, wrinkles, fit and interaction with the body.

SKIN TONE CONSISTENCY — IMPORTANT

Match the skin tone / undertone from the SECOND image (IMAGE 2 — identity reference) across the whole person, not only the face.

Any visible skin in the FIRST image must stay consistent with that reference: face, neck, ears, chest/décolletage if visible, arms, hands, fingers, legs, feet — same complexion family (e.g. if the reference face is olive/tan/deep, hands and arms must not look pale/white or a mismatched hue).

Keep natural variation from lighting and shadows in IMAGE 1, but do not leave a face–body skin-color mismatch.
""".strip()

_PHOTOREALISM_TAIL = """
PHOTOREALISM

The final result must look like a genuine photograph captured with a real camera.

No sticker effect.

No cutout effect.

No collage effect.

No obvious face-swap effect.

No oversized head.

No unnatural proportions.

No plastic skin.

No beauty-filter appearance.

No artificial facial features.

No distorted hands or fingers.

No anatomical errors.

Match the lighting, shadows, skin texture, fabric texture, perspective, depth of field, lens characteristics and image grain of the FIRST image (IMAGE 1 — photo to edit).

The replacement must be completely integrated into the scene.

The FIRST image (IMAGE 1 — photo to edit) controls the scene, pose, body position, clothing, expression, gaze, composition and proportions.

The SECOND image (IMAGE 2 — face / identity reference) controls the identity, recognizable appearance, and consistent skin tone of the person (including visible body skin).

Do not redesign the image. Do not reinterpret the pose. Reproduce the FIRST image as faithfully as possible while replacing only the person with the identity from the SECOND image.

Output a single edited version of IMAGE 1. Do not output IMAGE 2.
""".strip()


def build_character_replace_prompt(
    *,
    remove_overlay_text: bool = True,
    extra_user_instructions: str | None = None,
) -> str:
    """Build the full character-replacement instruction for the model."""
    text_section = _TEXT_REMOVAL_SECTION if remove_overlay_text else _TEXT_KEEP_SECTION
    parts = [_IMAGE_ROLES, _CORE_PROMPT, text_section, _PHOTOREALISM_TAIL]
    cleaned = (extra_user_instructions or "").strip()
    if cleaned:
        parts.append(
            "ADDITIONAL USER INSTRUCTIONS\n\n"
            "Apply these extra instructions without violating the identity/scene rules above:\n"
            f"{cleaned}"
        )
    return "\n\n".join(parts)


def build_user_prompt(
    *,
    remove_overlay_text: bool,
    preserve_original_composition: bool = True,
    preserve_pose: bool = True,
    preserve_lighting: bool = True,
    extra_user_instructions: str | None = None,
) -> str:
    """Alias kept for tests/call sites; preserve_* are already encoded in the core prompt."""
    _ = (preserve_original_composition, preserve_pose, preserve_lighting)
    return build_character_replace_prompt(
        remove_overlay_text=remove_overlay_text,
        extra_user_instructions=extra_user_instructions,
    )
