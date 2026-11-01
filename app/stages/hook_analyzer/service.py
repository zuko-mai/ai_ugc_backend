import asyncio
from pathlib import Path
from uuid import UUID

import cv2

from app.core.config import Settings
from app.core.logging import get_logger
from app.stages.hook_analyzer.audio_context import probe_audio_context
from app.stages.hook_analyzer.exceptions import SessionNotFoundError
from app.stages.hook_analyzer.frames import extract_frames
from app.stages.hook_analyzer.gemini_motion import generate_kling_motion_prompt
from app.stages.hook_analyzer.gemini_pricing import TokenUsage
from app.stages.hook_analyzer.motion import build_motion_plan
from app.stages.hook_analyzer.schemas import (
    CandidatesRequest,
    CandidatesResponse,
    FaceObservation,
    FrameCandidate,
    GeminiUsage,
    HookInterval,
    MotionPlanRequest,
    MotionPlanResponse,
    ReanalyzeRequest,
)
from app.stages.hook_analyzer.scoring import RankedFrame, observe_all, rank_frames
from app.stages.hook_analyzer.session import AnalysisSessionStore, make_analysis_key
from app.stages.hook_analyzer.trim import trim_video_clip
from app.stages.hook_analyzer.video_resolver import resolve_video_ref
from app.storage.base import FileStorage

logger = get_logger(__name__)


class HookAnalyzerService:
    def __init__(self, storage: FileStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self.sessions = AnalysisSessionStore(storage)

    async def find_candidates(self, request: CandidatesRequest) -> CandidatesResponse:
        interval = HookInterval(
            start_seconds=request.hook_start_seconds,
            end_seconds=request.hook_end_seconds,
        )
        analysis_key = make_analysis_key(interval.start_seconds, interval.end_seconds)
        video_id, video_path = await resolve_video_ref(
            video_id=request.video_id,
            video_url=request.video_url,
            storage=self.storage,
            settings=self.settings,
        )

        _observations, ranked = await self._ensure_analysis(
            video_id=video_id,
            video_path=video_path,
            interval=interval,
            analysis_key=analysis_key,
            candidate_count=request.candidate_count,
        )

        candidates = await self._persist_ranked_candidates(
            video_id=video_id,
            analysis_key=analysis_key,
            ranked=ranked,
        )
        self.sessions.save_candidates(video_id, analysis_key, candidates)

        logger.info(
            "Hook candidates ready video_id=%s count=%s key=%s",
            video_id,
            len(candidates),
            analysis_key,
        )
        return CandidatesResponse(
            video_id=video_id,
            candidates=candidates,
            analysis_key=analysis_key,
        )

    async def build_motion_plan(self, request: MotionPlanRequest) -> MotionPlanResponse:
        interval = HookInterval(
            start_seconds=request.start_seconds,
            end_seconds=request.end_seconds,
        )
        analysis_key = make_analysis_key(interval.start_seconds, interval.end_seconds)

        video_id, video_path = await resolve_video_ref(
            video_id=request.video_id,
            video_url=request.video_url,
            storage=self.storage,
            settings=self.settings,
        )

        try:
            observations = self.sessions.load_observations(video_id, analysis_key)
            audio_context = self.sessions.load_audio_context(video_id, analysis_key)
            logger.info(
                "Motion observations from cache video_id=%s key=%s",
                video_id,
                analysis_key,
            )
        except SessionNotFoundError:
            observations, _ranked = await self._ensure_analysis(
                video_id=video_id,
                video_path=video_path,
                interval=interval,
                analysis_key=analysis_key,
                candidate_count=5,
            )
            audio_context = self.sessions.load_audio_context(video_id, analysis_key)

        trimmed_path, trimmed_video_url = await self._ensure_trimmed_clip(
            video_id=video_id,
            video_path=video_path,
            analysis_key=analysis_key,
            start_seconds=request.start_seconds,
            duration_seconds=request.duration_seconds,
        )
        gemini_result = await generate_kling_motion_prompt(
            video_path=trimmed_path,
            api_key=self.settings.google_ai_api_key,
            model=self.settings.hook_analyzer_motion_model,
            start_seconds=request.start_seconds,
            duration_seconds=request.duration_seconds,
        )

        return build_motion_plan(
            video_id=video_id,
            interval=interval,
            observations=observations,
            audio_context=audio_context,
            analysis_key=analysis_key,
            motion_prompt=gemini_result.motion_prompt,
            trimmed_video_url=trimmed_video_url,
            gemini_usage=_to_gemini_usage(gemini_result.usage),
        )

    async def reanalyze(self, request: ReanalyzeRequest) -> MotionPlanResponse:
        """Recompute motion plan after frame selection using cached observations.

        Creates the trimmed clip if missing (candidates-only flow never trims),
        then always regenerates motionPrompt via Gemini.
        """
        interval = HookInterval(
            start_seconds=request.start_seconds,
            end_seconds=request.end_seconds,
        )
        analysis_key = make_analysis_key(interval.start_seconds, interval.end_seconds)

        selected = self.sessions.require_frame(
            request.video_id, analysis_key, request.selected_frame_id
        )
        observations = self.sessions.load_observations(request.video_id, analysis_key)
        audio_context = self.sessions.load_audio_context(request.video_id, analysis_key)

        _, video_path = await resolve_video_ref(
            video_id=request.video_id,
            video_url=None,
            storage=self.storage,
            settings=self.settings,
        )
        trimmed_path, trimmed_video_url = await self._ensure_trimmed_clip(
            video_id=request.video_id,
            video_path=video_path,
            analysis_key=analysis_key,
            start_seconds=request.start_seconds,
            duration_seconds=request.duration_seconds,
        )
        gemini_result = await generate_kling_motion_prompt(
            video_path=trimmed_path,
            api_key=self.settings.google_ai_api_key,
            model=self.settings.hook_analyzer_motion_model,
            start_seconds=request.start_seconds,
            duration_seconds=request.duration_seconds,
        )

        plan = build_motion_plan(
            video_id=request.video_id,
            interval=interval,
            observations=observations,
            audio_context=audio_context,
            selected_frame_id=selected.frame_id,
            analysis_key=analysis_key,
            motion_prompt=gemini_result.motion_prompt,
            trimmed_video_url=trimmed_video_url,
            gemini_usage=_to_gemini_usage(gemini_result.usage),
        )
        logger.info(
            "Reanalyzed motion plan video_id=%s selected_frame=%s ts=%.3f "
            "quality=%.3f",
            request.video_id,
            selected.frame_id,
            selected.timestamp_seconds,
            selected.quality_score,
        )
        return plan

    async def _ensure_trimmed_clip(
        self,
        *,
        video_id: UUID,
        video_path: Path,
        analysis_key: str,
        start_seconds: float,
        duration_seconds: float,
    ) -> tuple[Path, str]:
        dest = self._trimmed_clip_path(video_id, analysis_key)
        if not dest.is_file() or dest.stat().st_size == 0:
            await trim_video_clip(
                video_path,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                output_path=dest,
            )
        relative_key = str(dest.relative_to(self.sessions._root())).replace("\\", "/")
        return dest, self.storage.get_url(relative_key)

    def _trimmed_clip_path(self, video_id: UUID, analysis_key: str) -> Path:
        return self.sessions.dir_for(video_id, analysis_key) / "trimmed_hook.mp4"

    async def _ensure_analysis(
        self,
        *,
        video_id: UUID,
        video_path: Path,
        interval: HookInterval,
        analysis_key: str,
        candidate_count: int,
    ) -> tuple[list[FaceObservation], list[RankedFrame]]:
        frames = await extract_frames(
            video_path,
            start_seconds=interval.start_seconds,
            end_seconds=interval.end_seconds,
            sample_fps=8.0,
            output_dir=self.sessions.dir_for(video_id, analysis_key) / "samples",
        )

        observations: list[FaceObservation] | None = None
        try:
            cached = self.sessions.load_observations(video_id, analysis_key)
            if len(cached) == len(frames) and all(
                abs(obs.timestamp_seconds - frame.timestamp_seconds) < 1e-3
                for obs, frame in zip(cached, frames, strict=True)
            ):
                observations = cached
                logger.info(
                    "Reusing cached observations video_id=%s key=%s count=%s",
                    video_id,
                    analysis_key,
                    len(cached),
                )
        except SessionNotFoundError:
            observations = None

        if observations is None:
            observations = await asyncio.to_thread(observe_all, frames)
            self.sessions.save_observations(video_id, analysis_key, observations)
            audio_context = await probe_audio_context(video_path)
            self.sessions.save_audio_context(video_id, analysis_key, audio_context)

        ranked = rank_frames(frames, observations, candidate_count=candidate_count)
        return observations, ranked

    async def _persist_ranked_candidates(
        self,
        *,
        video_id: UUID,
        analysis_key: str,
        ranked: list[RankedFrame],
    ) -> list[FrameCandidate]:
        candidates: list[FrameCandidate] = []
        root = self.sessions._root()
        for item in ranked:
            dest = (
                self.sessions.dir_for(video_id, analysis_key)
                / "candidates"
                / f"{item.frame_id}.jpg"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            ok = await asyncio.to_thread(
                cv2.imwrite,
                str(dest),
                item.frame.image_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            if not ok:
                logger.warning("Failed to write candidate image %s", dest)
            relative_key = str(dest.relative_to(root)).replace("\\", "/")
            candidates.append(
                FrameCandidate(
                    frame_id=item.frame_id,
                    timestamp_seconds=item.frame.timestamp_seconds,
                    image_url=self.storage.get_url(relative_key),
                    quality_score=item.quality_score,
                    reason=item.reason,
                )
            )
        return candidates


def _to_gemini_usage(usage: TokenUsage) -> GeminiUsage:
    return GeminiUsage(
        model=usage.model,
        prompt_tokens=usage.prompt_tokens,
        candidates_tokens=usage.candidates_tokens,
        thoughts_tokens=usage.thoughts_tokens,
        total_tokens=usage.total_tokens,
        input_cost_usd=usage.input_cost_usd,
        output_cost_usd=usage.output_cost_usd,
        estimated_cost_usd=usage.estimated_cost_usd,
        currency=usage.currency,
        pricing_note=usage.pricing_note,
    )
