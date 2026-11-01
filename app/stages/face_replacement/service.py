"""Face Replacement stage — first-frame identity swap for UGC image-to-video."""

from __future__ import annotations

import time
from uuid import UUID, uuid4

from fastapi import UploadFile

from app.core.config import Settings
from app.core.logging import get_logger
from app.stages.face_replacement.exceptions import (
    FrameNotSuitableError,
    ImageFetchError,
    MissingFaceReferenceError,
)
from app.stages.face_replacement.images import LoadedImage, load_image, load_uploaded_image
from app.stages.face_replacement.models import (
    DEFAULT_FACE_REPLACEMENT_MODEL,
    resolve_model_name,
)
from app.stages.face_replacement.pricing import empty_usage
from app.stages.face_replacement.prompt import build_character_replace_prompt
from app.stages.face_replacement.providers.base import (
    FaceReplaceProviderRequest,
    ProviderImage,
)
from app.stages.face_replacement.providers.registry import get_provider
from app.stages.face_replacement.schemas import (
    FaceReplacementRequest,
    FaceReplacementResponse,
    FaceRerunRequest,
    GenerationRecord,
    GenerationUsage,
)
from app.stages.face_replacement.store import FaceReplacementStore
from app.stages.face_replacement.text_check import check_residual_overlay_text
from app.storage.base import FileStorage

logger = get_logger(__name__)


class FaceReplacementService:
    def __init__(self, storage: FileStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self.store = FaceReplacementStore(storage)

    async def replace_face(
        self,
        request: FaceReplacementRequest,
        *,
        face_reference: UploadFile | None = None,
    ) -> FaceReplacementResponse:
        face = await self._resolve_face_reference(
            upload=face_reference,
            face_reference_url=request.face_reference_url,
        )
        return await self._generate(
            video_id=request.video_id,
            video_url=request.video_url,
            selected_frame_url=request.selected_frame_url,
            face=face,
            face_reference_filename=(
                face_reference.filename if face_reference is not None else None
            ),
            remove_overlay_text=request.remove_overlay_text,
            preserve_original_composition=request.preserve_original_composition,
            preserve_pose=request.preserve_pose,
            preserve_lighting=request.preserve_lighting,
            model_provider=request.model_provider,
            model_name=request.model_name,
            user_prompt=request.user_prompt,
        )

    async def get_generation(self, generation_id: UUID) -> FaceReplacementResponse:
        record = await self.store.get_generation(generation_id)
        return self._to_response(record)

    async def rerun(
        self,
        generation_id: UUID,
        override: FaceRerunRequest | None = None,
    ) -> FaceReplacementResponse:
        previous = await self.store.get_generation(generation_id)
        if not previous.face_reference_key:
            raise MissingFaceReferenceError(
                "Stored generation has no archived face reference to reuse",
                details={"generation_id": str(generation_id)},
            )
        face_path = self.storage.get_local_path(previous.face_reference_key)
        face = await load_uploaded_image(
            data=face_path.read_bytes(),
            filename=face_path.name,
            content_type=None,
            settings=self.settings,
            role="face_reference",
            min_width=self.settings.min_face_reference_width,
            min_height=self.settings.min_face_reference_height,
        )
        override = override or FaceRerunRequest()
        return await self._generate(
            video_id=previous.video_id,
            video_url=previous.video_url,
            selected_frame_url=previous.selected_frame_url,
            face=face,
            face_reference_filename=previous.face_reference_filename,
            remove_overlay_text=(
                previous.text_removal_requested
                if override.remove_overlay_text is None
                else override.remove_overlay_text
            ),
            preserve_original_composition=(
                previous.preserve_original_composition
                if override.preserve_original_composition is None
                else override.preserve_original_composition
            ),
            preserve_pose=(
                previous.preserve_pose
                if override.preserve_pose is None
                else override.preserve_pose
            ),
            preserve_lighting=(
                previous.preserve_lighting
                if override.preserve_lighting is None
                else override.preserve_lighting
            ),
            model_provider=override.model_provider or previous.model_provider,
            model_name=override.model_name or previous.model_name,
            user_prompt=(
                previous.user_prompt
                if override.user_prompt is None
                else override.user_prompt
            ),
            face_reference_key=previous.face_reference_key,
        )

    async def _generate(
        self,
        *,
        video_id: UUID,
        video_url: str | None,
        selected_frame_url: str,
        face: LoadedImage,
        face_reference_filename: str | None,
        remove_overlay_text: bool,
        preserve_original_composition: bool,
        preserve_pose: bool,
        preserve_lighting: bool,
        model_provider: str,
        model_name: str,
        user_prompt: str | None,
        face_reference_key: str | None = None,
    ) -> FaceReplacementResponse:
        started = time.perf_counter()
        provider_name = (
            model_provider or self.settings.face_replacement_provider or "gemini"
        ).strip().lower()
        resolved_model_name = resolve_model_name(
            model_name
            or self.settings.face_replacement_model
            or DEFAULT_FACE_REPLACEMENT_MODEL
        )

        base = await load_image(
            selected_frame_url,
            settings=self.settings,
            storage=self.storage,
            role="selected_frame",
            min_width=self.settings.min_selected_frame_width,
            min_height=self.settings.min_selected_frame_height,
        )

        await self.store.archive_input(video_id, "selected_frame", base)
        face_key = face_reference_key or await self.store.archive_input(
            video_id, "face_reference", face
        )

        compiled_prompt = build_character_replace_prompt(
            remove_overlay_text=remove_overlay_text,
            extra_user_instructions=user_prompt,
        )
        provider = get_provider(provider_name, self.settings)
        provider_result = await provider.replace_face(
            FaceReplaceProviderRequest(
                base_frame=_to_provider_image(base),
                face_reference=_to_provider_image(face),
                remove_overlay_text=remove_overlay_text,
                preserve_original_composition=preserve_original_composition,
                preserve_pose=preserve_pose,
                preserve_lighting=preserve_lighting,
                model_name=resolved_model_name,
                timeout_seconds=float(self.settings.face_replacement_timeout_seconds),
                extra_user_instructions=user_prompt,
            )
        )

        warnings: list[str] = list(provider_result.provider_warnings)

        readiness_warnings = _check_first_frame_readiness(
            width=provider_result.width,
            height=provider_result.height,
            base_width=base.width,
            base_height=base.height,
            image_bytes=provider_result.image_bytes,
            min_width=self.settings.min_selected_frame_width,
            min_height=self.settings.min_selected_frame_height,
        )
        hard_failures = [w for w in readiness_warnings if w.startswith("fatal:")]
        if hard_failures:
            raise FrameNotSuitableError(
                "Generated frame is not suitable as an image-to-video first frame",
                details={"reasons": hard_failures},
            )
        warnings.extend(w for w in readiness_warnings if not w.startswith("fatal:"))

        if remove_overlay_text:
            text_check = await check_residual_overlay_text(
                provider_result.image_bytes,
                api_key=self.settings.google_ai_api_key,
                model=self.settings.gemini_model,
            )
            if text_check.residual_text_likely:
                warnings.append(
                    "residual_overlay_text_detected: "
                    f"method={text_check.method} confidence={text_check.confidence:.2f}"
                    + (f" ({text_check.details})" if text_check.details else "")
                )
                logger.warning(
                    "Residual overlay text suspected video_id=%s method=%s conf=%.2f",
                    video_id,
                    text_check.method,
                    text_check.confidence,
                )

        resolved_model = provider_result.model_name or resolved_model_name
        token_usage = provider_result.usage or empty_usage(model=resolved_model)
        processing_time_seconds = round(time.perf_counter() - started, 4)
        usage = GenerationUsage(
            prompt_tokens=token_usage.prompt_tokens,
            candidates_tokens=token_usage.candidates_tokens,
            thoughts_tokens=token_usage.thoughts_tokens,
            total_tokens=token_usage.total_tokens,
            input_cost_usd=token_usage.input_cost_usd,
            output_cost_usd=token_usage.output_cost_usd,
            estimated_cost_usd=token_usage.estimated_cost_usd,
            currency=token_usage.currency,
            processing_time_seconds=processing_time_seconds,
            provider_latency_seconds=provider_result.latency_seconds,
            pricing_note=token_usage.pricing_note or None,
        )

        generation_id = uuid4()
        record = GenerationRecord(
            generation_id=generation_id,
            video_id=video_id,
            video_url=video_url,
            selected_frame_url=selected_frame_url,
            face_reference_filename=face_reference_filename,
            face_reference_key=face_key,
            face_reference_url=self.storage.get_url(face_key),
            generated_frame_key="",
            width=provider_result.width,
            height=provider_result.height,
            model_provider=provider_name,
            model_name=resolved_model,
            text_removal_requested=remove_overlay_text,
            preserve_original_composition=preserve_original_composition,
            preserve_pose=preserve_pose,
            preserve_lighting=preserve_lighting,
            user_prompt=user_prompt,
            compiled_prompt=compiled_prompt,
            warnings=warnings,
            usage=usage,
        )
        image_key, generation_id = await self.store.save_generation(
            video_id=video_id,
            image_bytes=provider_result.image_bytes,
            mime_type=provider_result.mime_type,
            record=record,
        )
        record.generated_frame_key = image_key
        record.generation_id = generation_id

        logger.info(
            "Face replacement done video_id=%s generation_id=%s provider=%s model=%s "
            "tokens=%s cost_usd=%s processing_s=%.3f warnings=%s",
            video_id,
            generation_id,
            provider_name,
            resolved_model,
            usage.total_tokens,
            usage.estimated_cost_usd,
            usage.processing_time_seconds,
            len(warnings),
        )
        return self._to_response(record)

    def _to_response(self, record: GenerationRecord) -> FaceReplacementResponse:
        if not record.usage:
            raise ImageFetchError(
                "Stored generation is missing usage metadata",
                details={"generation_id": str(record.generation_id)},
            )
        return FaceReplacementResponse(
            video_id=record.video_id,
            generated_frame_url=self.storage.get_url(record.generated_frame_key),
            width=record.width,
            height=record.height,
            model_provider=record.model_provider,
            model_name=record.model_name,
            text_removal_requested=record.text_removal_requested,
            status="success",
            warnings=record.warnings,
            generation_id=record.generation_id,
            usage=record.usage,
        )

    async def _resolve_face_reference(
        self,
        *,
        upload: UploadFile | None,
        face_reference_url: str | None,
    ) -> LoadedImage:
        if upload is not None and upload.filename:
            data = await upload.read()
            if data:
                return await load_uploaded_image(
                    data=data,
                    filename=upload.filename,
                    content_type=upload.content_type,
                    settings=self.settings,
                    role="face_reference",
                    min_width=self.settings.min_face_reference_width,
                    min_height=self.settings.min_face_reference_height,
                )
        if face_reference_url:
            return await load_image(
                face_reference_url,
                settings=self.settings,
                storage=self.storage,
                role="face_reference",
                min_width=self.settings.min_face_reference_width,
                min_height=self.settings.min_face_reference_height,
            )
        raise MissingFaceReferenceError(
            "Provide multipart face_reference or face_reference_url",
        )


def _to_provider_image(image: LoadedImage) -> ProviderImage:
    return ProviderImage(
        data=image.data,
        mime_type=image.mime_type,
        width=image.width,
        height=image.height,
        source_url=image.source_url,
    )


def _check_first_frame_readiness(
    *,
    width: int,
    height: int,
    base_width: int,
    base_height: int,
    image_bytes: bytes,
    min_width: int,
    min_height: int,
) -> list[str]:
    warnings: list[str] = []
    if not image_bytes:
        warnings.append("fatal: empty_image")
        return warnings
    if width < min_width or height < min_height:
        warnings.append(
            f"fatal: resolution_too_low {width}x{height} < {min_width}x{min_height}"
        )
    if base_width > 0 and base_height > 0:
        base_ar = base_width / base_height
        out_ar = width / max(height, 1)
        if abs(base_ar - out_ar) / base_ar > 0.08:
            warnings.append(
                f"aspect_ratio_changed: base={base_ar:.3f} generated={out_ar:.3f}"
            )
        if abs(width - base_width) / base_width > 0.25 or abs(height - base_height) / base_height > 0.25:
            warnings.append(
                f"dimensions_changed: base={base_width}x{base_height} "
                f"generated={width}x{height}"
            )
    return warnings
