from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import UploadFile

from app.core.config import Settings
from app.pipeline.exceptions import PipelineError, PipelineOrderError
from app.pipeline.schemas import (
    AiCostRun,
    CandidateOptions,
    CompositionState,
    FaceState,
    HookAnalysis,
    HookTextOptions,
    HookTextState,
    HookVideoOptions,
    HookVideoState,
    MusicState,
    PipelineSession,
    Product,
    RenderOptions,
    RenderState,
    SelectFrameRequest,
    SelectTextRequest,
    SourceVideo,
)
from app.pipeline.store import AssetStore, SessionStore
from app.stages.face_replacement.schemas import FaceReplacementRequest
from app.stages.face_replacement.service import FaceReplacementService
from app.stages.hook_analyzer.schemas import CandidatesRequest, ReanalyzeRequest
from app.stages.hook_analyzer.service import HookAnalyzerService
from app.stages.hook_text_engine.schemas import SelectRequest
from app.stages.hook_text_engine.service import HookTextEngineService
from app.stages.hook_video_generator.schemas import HookVideoGenerateRequest
from app.stages.hook_video_generator.service import HookVideoService
from app.stages.music_overlay.schemas import OverlaySettings, VideoAssetRef as MusicVideoRef
from app.stages.music_overlay.service import MusicOverlayService
from app.stages.product_demo_composer.schemas import (
    ComposeSettings,
    VideoAssetRef as ComposeVideoRef,
)
from app.stages.product_demo_composer.service import ProductDemoComposerService
from app.stages.text_renderer.schemas import TextRenderRequest
from app.stages.text_renderer.service import TextRendererService
from app.stages.video_ingestion.service import VideoIngestionService
from app.storage.base import FileStorage


class PipelineService:
    def __init__(self, storage: FileStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self.sessions = SessionStore(storage)
        self.assets = AssetStore(storage)

    async def create_session(self) -> PipelineSession:
        return await self.sessions.create()

    async def get_session(self, session_id: UUID) -> PipelineSession:
        return await self.sessions.get(session_id)

    async def set_product(self, session_id: UUID, product: Product) -> PipelineSession:
        session = await self.sessions.get(session_id)
        session.product = product
        session.source_video = None
        session.hook_analysis = None
        session.face = None
        session.hook_video = None
        session.hook_text = None
        session.render = None
        session.composition = None
        session.music = None
        session.step = 0
        return await self.sessions.save(session)

    async def ingest_upload(
        self, session_id: UUID, upload: UploadFile
    ) -> PipelineSession:
        session = await self.sessions.get(session_id)
        self._require(session.product, 1, "product")
        result = await VideoIngestionService(self.storage, self.settings).ingest_upload(upload)
        return await self._set_source(session, result.video_id, result.video_url)

    async def ingest_url(self, session_id: UUID, url: str) -> PipelineSession:
        session = await self.sessions.get(session_id)
        self._require(session.product, 1, "product")
        result = await VideoIngestionService(self.storage, self.settings).ingest_url(url)
        return await self._set_source(session, result.video_id, result.video_url)

    async def find_candidates(
        self, session_id: UUID, options: CandidateOptions
    ) -> PipelineSession:
        session = await self.sessions.get(session_id)
        self._require(session.source_video, 2, "source_video")
        source = session.source_video
        result = await HookAnalyzerService(self.storage, self.settings).find_candidates(
            CandidatesRequest(
                video_id=source.video_id,
                video_url=source.video_url,
                **options.model_dump(),
            )
        )
        session.hook_analysis = HookAnalysis(
            analysis_key=result.analysis_key,
            hook_start_seconds=options.hook_start_seconds,
            hook_end_seconds=options.hook_end_seconds,
            candidates=[
                item.model_dump(mode="json") for item in result.candidates
            ],
        )
        session.face = None
        session.hook_video = None
        session.hook_text = None
        session.render = None
        session.composition = None
        session.music = None
        session.step = 2
        return await self.sessions.save(session)

    async def select_frame(
        self, session_id: UUID, body: SelectFrameRequest
    ) -> PipelineSession:
        session = await self.sessions.get(session_id)
        analysis = session.hook_analysis
        self._require(analysis and analysis.candidates, 2, "hook candidates")
        match = next(
            (
                item
                for item in analysis.candidates or []
                if str(item.get("frame_id")) == str(body.selected_frame_id)
            ),
            None,
        )
        if match is None:
            raise PipelineError(
                "selected_frame_id is not present in session candidates",
                details={"selected_frame_id": str(body.selected_frame_id)},
            )
        analysis.selected_frame_id = body.selected_frame_id
        analysis.selected_frame_url = str(match["image_url"])
        if body.generate_motion_plan:
            source = session.source_video
            assert source is not None
            start_seconds, duration_seconds = self._motion_window_for_analysis(
                analysis, body
            )
            plan = await HookAnalyzerService(self.storage, self.settings).reanalyze(
                ReanalyzeRequest(
                    video_id=source.video_id,
                    selected_frame_id=body.selected_frame_id,
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                )
            )
            if not plan.motionPrompt:
                raise PipelineError(
                    "motionPrompt was empty after reanalyze; check GOOGLE_AI_API_KEY",
                    details={"video_id": str(source.video_id)},
                )
            analysis.motion_prompt = plan.motionPrompt
            analysis.trimmed_video_url = plan.trimmed_video_url
            usage = plan.gemini_usage
            analysis.ai_runs = list(analysis.ai_runs) + [
                AiCostRun(
                    label="motionPrompt",
                    model=usage.model if usage else self.settings.hook_analyzer_motion_model,
                    cost_usd=usage.estimated_cost_usd if usage else None,
                    is_rerun=any(r.label == "motionPrompt" for r in analysis.ai_runs),
                )
            ]
        session.hook_analysis = analysis
        return await self.sessions.save(session)

    async def generate_face(
        self,
        session_id: UUID,
        *,
        face_file: UploadFile | None,
        face_asset_id: UUID | None,
        user_prompt: str | None,
    ) -> PipelineSession:
        session = await self.sessions.get(session_id)
        analysis = session.hook_analysis
        self._require(analysis and analysis.selected_frame_url, 3, "selected hook frame")
        source = session.source_video
        assert source is not None
        asset = None
        upload = face_file
        if upload is not None:
            asset = await self.assets.upload(upload, "face_reference")
            asset, upload = await self.assets.as_upload(asset.asset_id)
        elif face_asset_id is not None:
            asset, upload = await self.assets.as_upload(face_asset_id)
        if upload is None:
            raise PipelineError("Provide face_file or face_asset_id")
        try:
            result = await FaceReplacementService(
                self.storage, self.settings
            ).replace_face(
                FaceReplacementRequest(
                    video_id=source.video_id,
                    video_url=source.video_url,
                    selected_frame_url=analysis.selected_frame_url,
                    user_prompt=user_prompt,
                    model_provider=self.settings.face_replacement_provider,
                    model_name=self.settings.face_replacement_model,
                ),
                face_reference=upload,
            )
        finally:
            await upload.close()
        prev_runs = list(session.face.ai_runs) if session.face else []
        session.face = FaceState(
            face_asset_id=asset.asset_id if asset else face_asset_id,
            face_url=asset.url if asset else None,
            generation_id=result.generation_id,
            generated_frame_url=result.generated_frame_url,
            user_prompt=user_prompt,
            ai_runs=prev_runs
            + [
                AiCostRun(
                    label="face_replacement",
                    model=result.model_name,
                    cost_usd=result.usage.estimated_cost_usd,
                    is_rerun=bool(prev_runs),
                )
            ],
        )
        session.step = 3
        return await self.sessions.save(session)

    async def generate_hook_video(
        self, session_id: UUID, options: HookVideoOptions
    ) -> PipelineSession:
        session = await self.sessions.get(session_id)
        self._require(
            session.face and session.face.generated_frame_url,
            4,
            "generated face frame",
        )
        source = session.source_video
        assert source is not None
        motion_prompt = options.prompt or (
            session.hook_analysis.motion_prompt if session.hook_analysis else None
        )
        if not motion_prompt:
            raise PipelineOrderError(
                "Hook video requires a motion prompt; provide prompt or generate a motion plan",
                details={"required_step": 2},
            )
        request_data = options.model_dump(exclude={"prompt"}, exclude_none=True)
        request_data["model_provider"] = request_data.get(
            "model_provider", self.settings.hook_video_provider
        )
        request_data["model_name"] = request_data.get(
            "model_name", self.settings.hook_video_model
        )
        accepted = await HookVideoService(self.storage, self.settings).submit(
            HookVideoGenerateRequest(
                video_id=source.video_id,
                first_frame_url=session.face.generated_frame_url,
                motion_plan={"motionPrompt": motion_prompt},
                **request_data,
            )
        )
        prev_runs = list(session.hook_video.ai_runs) if session.hook_video else []
        provider = request_data.get("model_provider") or self.settings.hook_video_provider
        model = request_data.get("model_name") or self.settings.hook_video_model
        session.hook_video = HookVideoState(
            job_id=accepted.job_id,
            motion_prompt=motion_prompt,
            video_url=session.hook_video.video_url if session.hook_video else None,
            asset_id=session.hook_video.asset_id if session.hook_video else None,
            ai_runs=prev_runs
            + [
                AiCostRun(
                    label="hook_video",
                    model=f"{provider}/{model}",
                    cost_usd=None,
                    is_rerun=bool(prev_runs),
                )
            ],
        )
        session.step = 4
        return await self.sessions.save(session)

    async def refresh_hook_video(self, session_id: UUID) -> PipelineSession:
        session = await self.sessions.get(session_id)
        self._require(
            session.hook_video and session.hook_video.job_id,
            4,
            "hook video job",
        )
        job = await HookVideoService(self.storage, self.settings).get_job(
            session.hook_video.job_id
        )
        if job.status == "success":
            session.hook_video.video_url = job.generated_video_url
            session.hook_video.asset_id = session.hook_video.asset_id or uuid4()
        return await self.sessions.save(session)

    async def extract_adapt_text(
        self, session_id: UUID, options: HookTextOptions
    ) -> PipelineSession:
        session = await self.sessions.get(session_id)
        self._require(
            session.hook_video and session.hook_video.video_url,
            5,
            "completed hook video",
        )
        source = session.source_video
        product = session.product
        assert source is not None and product is not None
        service = HookTextEngineService(self.storage, self.settings)
        extraction, adaptation, cost_usd = await service.extract_and_adapt(
            video_id=source.video_id,
            video_url=source.video_url,
            start_seconds=options.start_seconds,
            end_seconds=options.end_seconds,
            languages=[product.language or "en"],
            product_name=product.name or "Product",
            product_description=product.description,
            target_audience=product.target_audience or "customers",
            language=product.language or "en",
            hook_goal=options.hook_goal,
            variants_count=options.variants_count,
        )
        prev_runs = list(session.hook_text.ai_runs) if session.hook_text else []
        session.hook_text = HookTextState(
            extraction_id=extraction.extraction_id,
            adaptation_id=adaptation.adaptation_id,
            variants=[item.model_dump(mode="json") for item in adaptation.variants],
            selected_text=session.hook_text.selected_text if session.hook_text else None,
            ai_runs=prev_runs
            + [
                AiCostRun(
                    label="hook_text_combined",
                    model=self.settings.hook_text_model,
                    cost_usd=cost_usd,
                    is_rerun=bool(prev_runs),
                )
            ],
        )
        session.step = 5
        return await self.sessions.save(session)

    async def select_text(
        self, session_id: UUID, body: SelectTextRequest
    ) -> PipelineSession:
        session = await self.sessions.get(session_id)
        self._require(
            session.hook_text and session.hook_text.adaptation_id,
            5,
            "hook text adaptation",
        )
        result = await HookTextEngineService(self.storage, self.settings).select(
            SelectRequest(
                adaptation_id=session.hook_text.adaptation_id,
                **body.model_dump(),
            )
        )
        session.hook_text.selected_text = result.selected_text
        return await self.sessions.save(session)

    async def render_text(
        self, session_id: UUID, options: RenderOptions
    ) -> PipelineSession:
        session = await self.sessions.get(session_id)
        self._require(
            session.hook_text and session.hook_text.selected_text,
            6,
            "selected hook text",
        )
        self._require(
            session.hook_video and session.hook_video.video_url,
            6,
            "completed hook video",
        )
        source = session.source_video
        assert source is not None
        result = await TextRendererService(self.storage, self.settings).render(
            TextRenderRequest(
                video_id=source.video_id,
                generated_video_url=session.hook_video.video_url,
                text=session.hook_text.selected_text,
                language=session.product.language or "en",
                output_format=options.output_format,
                skip_text=options.skip_text,
            )
        )
        session.render = RenderState(
            render_id=result.render_id,
            asset_id=result.asset_id or result.render_id,
            video_url=result.rendered_video_url,
        )
        session.step = 6
        return await self.sessions.save(session)

    async def compose(
        self,
        session_id: UUID,
        *,
        demo_file: UploadFile | None,
        demo_asset_id: UUID | None,
    ) -> PipelineSession:
        session = await self.sessions.get(session_id)
        self._require(session.render and session.render.video_url, 7, "rendered hook video")
        upload = demo_file
        asset = None
        if upload is not None:
            asset = await self.assets.upload(upload, "product_demo")
            asset, upload = await self.assets.as_upload(asset.asset_id)
        elif demo_asset_id is not None:
            asset, upload = await self.assets.as_upload(demo_asset_id)
        if upload is None or asset is None:
            raise PipelineError("Provide demo_file or demo_asset_id")
        try:
            accepted = await ProductDemoComposerService(
                self.storage, self.settings
            ).submit(
                project_id=session.session_id,
                hook_video=ComposeVideoRef(
                    asset_id=session.render.asset_id or uuid4(),
                    video_url=session.render.video_url,
                ),
                demo_video=upload,
                demo_asset_id=asset.asset_id,
                settings=ComposeSettings(),
            )
        finally:
            await upload.close()
        session.composition = CompositionState(
            job_id=accepted.job_id,
            demo_asset_id=asset.asset_id,
        )
        session.step = 7
        return await self.sessions.save(session)

    async def refresh_composition(self, session_id: UUID) -> PipelineSession:
        session = await self.sessions.get(session_id)
        self._require(
            session.composition and session.composition.job_id,
            7,
            "composition job",
        )
        job = await ProductDemoComposerService(self.storage, self.settings).get_job(
            session.composition.job_id
        )
        if job.status == "success":
            session.composition.asset_id = job.asset_id
            session.composition.video_url = job.video_url
        return await self.sessions.save(session)

    async def add_music(
        self,
        session_id: UUID,
        *,
        music_file: UploadFile | None,
        music_asset_id: UUID | None,
        music_start_offset_seconds: float = 0.0,
        video_start_seconds: float = 0.0,
        music_volume: float = 0.3,
    ) -> PipelineSession:
        session = await self.sessions.get(session_id)
        self._require(
            session.composition and session.composition.video_url,
            8,
            "completed composition",
        )
        upload = music_file
        asset = None
        if upload is not None:
            asset = await self.assets.upload(upload, "music")
            asset, upload = await self.assets.as_upload(asset.asset_id)
        elif music_asset_id is not None:
            asset, upload = await self.assets.as_upload(music_asset_id)
        if upload is None or asset is None:
            raise PipelineError("Provide music_file or music_asset_id")
        try:
            accepted = await MusicOverlayService(self.storage, self.settings).submit_upload(
                project_id=session.session_id,
                video=MusicVideoRef(
                    asset_id=session.composition.asset_id or uuid4(),
                    video_url=session.composition.video_url,
                ),
                music=upload,
                music_asset_id=asset.asset_id,
                settings=OverlaySettings(
                    music_start_offset_seconds=music_start_offset_seconds,
                    video_start_seconds=video_start_seconds,
                    music_volume=music_volume,
                ),
            )
        finally:
            await upload.close()
        session.music = MusicState(
            job_id=accepted.job_id,
            music_asset_id=asset.asset_id,
        )
        session.step = 8
        return await self.sessions.save(session)

    async def refresh_music(self, session_id: UUID) -> PipelineSession:
        session = await self.sessions.get(session_id)
        self._require(session.music and session.music.job_id, 8, "music job")
        job = await MusicOverlayService(self.storage, self.settings).get_job(
            session.music.job_id
        )
        if job.status == "success":
            session.music.asset_id = job.asset_id
            session.music.video_url = job.video_url
        return await self.sessions.save(session)

    async def _set_source(
        self, session: PipelineSession, video_id: UUID, video_url: str
    ) -> PipelineSession:
        session.source_video = SourceVideo(video_id=video_id, video_url=video_url)
        session.hook_analysis = None
        session.face = None
        session.hook_video = None
        session.hook_text = None
        session.render = None
        session.composition = None
        session.music = None
        session.step = 1
        return await self.sessions.save(session)

    @staticmethod
    def _motion_window_for_analysis(
        analysis: HookAnalysis,
        body: SelectFrameRequest,
    ) -> tuple[float, float]:
        """Match reanalyze cache key to the candidates interval.

        Candidates are stored under analysis_key from hook_start/hook_end
        (default 0–4s). Reanalyze used to default to 0–3s and miss the cache.
        """
        if body.start_seconds is not None and body.duration_seconds is not None:
            return body.start_seconds, body.duration_seconds

        start = analysis.hook_start_seconds
        end = analysis.hook_end_seconds
        if start is None or end is None:
            if analysis.analysis_key and "_" in analysis.analysis_key:
                left, right = analysis.analysis_key.split("_", 1)
                start = float(left)
                end = float(right)
            else:
                start = 0.0
                end = 4.0
        duration = max(end - start, 0.1)
        if body.start_seconds is not None:
            start = body.start_seconds
            duration = max(end - start, 0.1)
        if body.duration_seconds is not None:
            duration = body.duration_seconds
        return start, duration

    @staticmethod
    def _require(value, step: int, field: str) -> None:  # noqa: ANN001
        if not value:
            raise PipelineOrderError(
                f"Pipeline step {step} requires {field}",
                details={"required_step": step - 1, "missing": field},
            )
