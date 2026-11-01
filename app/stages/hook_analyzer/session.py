import json
from pathlib import Path
from uuid import UUID

from app.core.logging import get_logger
from app.stages.hook_analyzer.exceptions import FrameNotFoundError, SessionNotFoundError
from app.stages.hook_analyzer.schemas import FaceObservation, FrameCandidate
from app.storage.base import FileStorage
from app.storage.local import LocalFileStorage

logger = get_logger(__name__)


def make_analysis_key(start_seconds: float, end_seconds: float) -> str:
    return f"{start_seconds:.3f}_{end_seconds:.3f}"


def analysis_prefix(video_id: UUID, analysis_key: str) -> str:
    return f"hooks/{video_id}/{analysis_key}"


class AnalysisSessionStore:
    """Persists extracted observations and candidates so reanalyze skips re-extraction."""

    def __init__(self, storage: FileStorage) -> None:
        self.storage = storage

    def _root(self) -> Path:
        if not isinstance(self.storage, LocalFileStorage):
            raise SessionNotFoundError(
                "Hook analyzer session cache currently requires LocalFileStorage"
            )
        return self.storage.root

    def dir_for(self, video_id: UUID, analysis_key: str) -> Path:
        path = self._root() / analysis_prefix(video_id, analysis_key)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_observations(
        self,
        video_id: UUID,
        analysis_key: str,
        observations: list[FaceObservation],
    ) -> None:
        path = self.dir_for(video_id, analysis_key) / "observations.json"
        payload = [o.model_dump() for o in observations]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Saved %s observations video_id=%s key=%s",
            len(observations),
            video_id,
            analysis_key,
        )

    def load_observations(
        self, video_id: UUID, analysis_key: str
    ) -> list[FaceObservation]:
        path = self.dir_for(video_id, analysis_key) / "observations.json"
        if not path.is_file():
            raise SessionNotFoundError(
                "No cached hook analysis for this video/interval. "
                "Call /candidates or /motion-plan first.",
                details={"video_id": str(video_id), "analysis_key": analysis_key},
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return [FaceObservation.model_validate(item) for item in data]

    def save_candidates(
        self,
        video_id: UUID,
        analysis_key: str,
        candidates: list[FrameCandidate],
    ) -> None:
        path = self.dir_for(video_id, analysis_key) / "candidates.json"
        path.write_text(
            json.dumps([c.model_dump(mode="json") for c in candidates], indent=2),
            encoding="utf-8",
        )

    def load_candidates(
        self, video_id: UUID, analysis_key: str
    ) -> list[FrameCandidate]:
        path = self.dir_for(video_id, analysis_key) / "candidates.json"
        if not path.is_file():
            raise SessionNotFoundError(
                "No cached candidates for this video/interval",
                details={"video_id": str(video_id), "analysis_key": analysis_key},
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return [FrameCandidate.model_validate(item) for item in data]

    def require_frame(
        self, video_id: UUID, analysis_key: str, frame_id: UUID
    ) -> FrameCandidate:
        candidates = self.load_candidates(video_id, analysis_key)
        for candidate in candidates:
            if candidate.frame_id == frame_id:
                return candidate
        raise FrameNotFoundError(
            "selected_frame_id is not among cached candidates",
            details={
                "video_id": str(video_id),
                "frame_id": str(frame_id),
                "analysis_key": analysis_key,
            },
        )

    def save_audio_context(
        self, video_id: UUID, analysis_key: str, audio_context: str | None
    ) -> None:
        path = self.dir_for(video_id, analysis_key) / "audio_context.json"
        path.write_text(json.dumps({"audio_context": audio_context}), encoding="utf-8")

    def load_audio_context(self, video_id: UUID, analysis_key: str) -> str | None:
        path = self.dir_for(video_id, analysis_key) / "audio_context.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("audio_context")
