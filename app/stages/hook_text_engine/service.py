import uuid
from typing import Literal
from uuid import UUID

from app.core.config import Settings
from app.core.logging import get_logger
from app.stages.hook_analyzer.trim import trim_video_clip
from app.stages.hook_text_engine.adaptation import generate_variants
from app.stages.hook_text_engine.asr import extract_spoken_text
from app.stages.hook_text_engine.audio import extract_audio_wav
from app.stages.hook_text_engine.combined import extract_and_adapt_combined
from app.stages.hook_text_engine.exceptions import AdaptationError, VariantNotFoundError
from app.stages.hook_text_engine.frames import extract_frames
from app.stages.hook_text_engine.ocr import extract_on_screen_text
from app.stages.hook_text_engine.schemas import (
    AdaptationRequest,
    AdaptationResponse,
    ExtractionRequest,
    ExtractionResponse,
    HookGoal,
    SelectRequest,
    SelectResponse,
    SessionResponse,
    TextSource,
)
from app.stages.hook_text_engine.store import HookTextSessionStore
from app.stages.hook_text_engine.video_resolver import resolve_video_ref
from app.storage.base import FileStorage

logger = get_logger(__name__)


class HookTextEngineService:
    def __init__(self, storage: FileStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self.sessions = HookTextSessionStore(storage)

    async def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        video_id, video_path = await resolve_video_ref(
            video_id=request.video_id,
            video_url=request.video_url,
            storage=self.storage,
            settings=self.settings,
        )
        extraction_id = uuid.uuid4()
        work_dir = self.sessions.work_dir(video_id, extraction_id)

        frames = await extract_frames(
            video_path,
            start_seconds=request.start_seconds,
            end_seconds=request.end_seconds,
            sample_fps=2.0,
            output_dir=work_dir / "frames",
        )
        media_model = self.settings.hook_text_model or self.settings.gemini_model
        on_screen = await extract_on_screen_text(
            frames,
            languages=request.languages,
            api_key=self.settings.google_ai_api_key,
            model=media_model,
        )

        audio_path = await extract_audio_wav(
            video_path,
            start_seconds=request.start_seconds,
            end_seconds=request.end_seconds,
            output_path=work_dir / "audio.wav",
        )
        spoken = await extract_spoken_text(
            audio_path,
            languages=request.languages,
            api_key=self.settings.google_ai_api_key,
            model=media_model,
            clip_start_seconds=request.start_seconds,
        )

        response = ExtractionResponse(
            video_id=video_id,
            extraction_id=extraction_id,
            start_seconds=request.start_seconds,
            end_seconds=request.end_seconds,
            on_screen_text=on_screen,
            spoken_text=spoken,
            languages=request.languages,
        )
        self.sessions.save_extraction(response)
        logger.info(
            "Hook text extraction done video_id=%s extraction_id=%s "
            "on_screen=%s spoken=%s",
            video_id,
            extraction_id,
            len(on_screen),
            len(spoken),
        )
        return response

    async def adapt(self, request: AdaptationRequest) -> AdaptationResponse:
        from app.stages.hook_text_engine.store import compose_original_from_extraction

        # Manual text always wins over auto-recognized / provided original.
        if request.manual_text and request.manual_text.strip():
            original = request.manual_text.strip()
            source: TextSource | None = "manual"
            if request.extraction_id is not None:
                self.sessions.load_extraction(request.extraction_id)
        elif request.original_text and request.original_text.strip():
            original = request.original_text.strip()
            source = request.preferred_source
            if request.extraction_id is not None:
                self.sessions.load_extraction(request.extraction_id)
        elif request.extraction_id is not None:
            extraction = self.sessions.load_extraction(request.extraction_id)
            original, source = compose_original_from_extraction(
                extraction,
                preferred_source=request.preferred_source,
            )
            if not original.strip():
                raise AdaptationError(
                    "Extraction has no usable hook text to adapt",
                    details={"extraction_id": str(request.extraction_id)},
                )
        else:
            raise AdaptationError(
                "Provide original_text, manual_text, or extraction_id",
            )

        adaptation_id = uuid.uuid4()

        if request.skip_adaptation:
            return self.sessions.save_adaptation(
                adaptation_id=adaptation_id,
                original_text=original,
                variants=[],
                selected_text=original,
                skip_adaptation=True,
                preferred_source=source,
                extraction_id=request.extraction_id,
            )

        variants = await generate_variants(
            original_text=original,
            product_name=request.product_name,
            product_description=request.product_description,
            target_audience=request.target_audience,
            language=request.language,
            hook_goal=request.hook_goal,
            variants_count=request.variants_count,
            api_key=self.settings.google_ai_api_key,
            model=self.settings.gemini_text_model or self.settings.gemini_model,
        )
        return self.sessions.save_adaptation(
            adaptation_id=adaptation_id,
            original_text=original,
            variants=variants,
            selected_text=None,
            skip_adaptation=False,
            preferred_source=source,
            extraction_id=request.extraction_id,
        )

    async def extract_and_adapt(
        self,
        *,
        video_id: UUID | None,
        video_url: str | None,
        start_seconds: float,
        end_seconds: float,
        languages: list[str],
        product_name: str,
        product_description: str,
        target_audience: str,
        language: str,
        hook_goal: HookGoal,
        variants_count: int,
    ) -> tuple[ExtractionResponse, AdaptationResponse, float | None]:
        """One Gemini video call for OCR + ASR + product variants (pipeline path)."""
        resolved_id, video_path = await resolve_video_ref(
            video_id=video_id,
            video_url=video_url,
            storage=self.storage,
            settings=self.settings,
        )
        extraction_id = uuid.uuid4()
        work_dir = self.sessions.work_dir(resolved_id, extraction_id)
        clip_path = work_dir / "hook_clip.mp4"
        duration = max(0.1, end_seconds - start_seconds)
        await trim_video_clip(
            video_path,
            start_seconds=start_seconds,
            duration_seconds=duration,
            output_path=clip_path,
        )

        combined = await extract_and_adapt_combined(
            video_path=clip_path,
            product_name=product_name,
            product_description=product_description,
            target_audience=target_audience,
            language=language,
            hook_goal=hook_goal,
            variants_count=variants_count,
            languages=languages,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            api_key=self.settings.google_ai_api_key,
            model=self.settings.hook_text_model,
            fallback_model=self.settings.hook_text_fallback_model,
        )

        extraction = ExtractionResponse(
            video_id=resolved_id,
            extraction_id=extraction_id,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            on_screen_text=combined.on_screen_text,
            spoken_text=combined.spoken_text,
            languages=languages,
        )
        self.sessions.save_extraction(extraction)

        adaptation_id = uuid.uuid4()
        adaptation = self.sessions.save_adaptation(
            adaptation_id=adaptation_id,
            original_text=combined.original_text,
            variants=combined.variants,
            selected_text=None,
            skip_adaptation=False,
            preferred_source=combined.preferred_source,
            extraction_id=extraction_id,
        )
        logger.info(
            "Combined hook-text done video_id=%s extraction_id=%s "
            "adaptation_id=%s model=%s cost_usd=%s",
            resolved_id,
            extraction_id,
            adaptation_id,
            combined.model,
            combined.estimated_cost_usd,
        )
        return extraction, adaptation, combined.estimated_cost_usd

    async def select(self, request: SelectRequest) -> SelectResponse:
        current = self.sessions.load_adaptation(request.adaptation_id)

        if request.manual_text is not None and request.manual_text.strip():
            selected = request.manual_text.strip()
            mode = "manual"
        elif request.variant_id is not None:
            match = next(
                (v for v in current.variants if v.variant_id == request.variant_id),
                None,
            )
            if match is None:
                raise VariantNotFoundError(
                    "variant_id not found in adaptation session",
                    details={
                        "adaptation_id": str(request.adaptation_id),
                        "variant_id": str(request.variant_id),
                    },
                )
            selected = match.hook_text
            mode = "variant"
        else:
            selected = current.original_text
            mode = "original"

        updated = self.sessions.update_selected(request.adaptation_id, selected)
        selection_mode: Literal["manual", "variant", "original"] = mode  # type: ignore[assignment]
        return SelectResponse(
            adaptation_id=updated.adaptation_id,
            original_text=updated.original_text,
            variants=updated.variants,
            selected_text=selected,
            selection_mode=selection_mode,
        )

    async def get_session(self, adaptation_id: UUID) -> SessionResponse:
        current = self.sessions.load_adaptation(adaptation_id)
        return SessionResponse(
            adaptation_id=current.adaptation_id,
            original_text=current.original_text,
            variants=current.variants,
            selected_text=current.selected_text,
            skip_adaptation=current.skip_adaptation,
            preferred_source=current.preferred_source,
            extraction_id=current.extraction_id,
        )
