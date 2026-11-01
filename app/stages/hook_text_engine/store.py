"""Persist extraction + adaptation sessions: original, variants, selected separately."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from app.core.logging import get_logger
from app.stages.hook_text_engine.exceptions import SessionNotFoundError
from app.stages.hook_text_engine.schemas import (
    AdaptationResponse,
    ExtractionResponse,
    HookVariant,
    TextSource,
)
from app.storage.base import FileStorage
from app.storage.local import LocalFileStorage

logger = get_logger(__name__)


class HookTextSessionStore:
    def __init__(self, storage: FileStorage) -> None:
        self.storage = storage

    def _root(self) -> Path:
        if not isinstance(self.storage, LocalFileStorage):
            raise SessionNotFoundError(
                "Hook Text Engine sessions currently require LocalFileStorage"
            )
        return self.storage.root

    def extraction_dir(self, extraction_id: UUID) -> Path:
        path = self._root() / "hook_text" / "extractions" / str(extraction_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def adaptation_dir(self, adaptation_id: UUID) -> Path:
        path = self._root() / "hook_text" / "adaptations" / str(adaptation_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def work_dir(self, video_id: UUID, extraction_id: UUID) -> Path:
        path = (
            self._root()
            / "hook_text"
            / str(video_id)
            / str(extraction_id)
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_extraction(self, response: ExtractionResponse) -> None:
        path = self.extraction_dir(response.extraction_id) / "extraction.json"
        path.write_text(
            json.dumps(response.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Saved extraction_id=%s video_id=%s on_screen=%s spoken=%s",
            response.extraction_id,
            response.video_id,
            len(response.on_screen_text),
            len(response.spoken_text),
        )

    def load_extraction(self, extraction_id: UUID) -> ExtractionResponse:
        path = self.extraction_dir(extraction_id) / "extraction.json"
        if not path.is_file():
            raise SessionNotFoundError(
                "Extraction session not found",
                details={"extraction_id": str(extraction_id)},
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExtractionResponse.model_validate(data)

    def save_adaptation(
        self,
        *,
        adaptation_id: UUID,
        original_text: str,
        variants: list[HookVariant],
        selected_text: str | None,
        skip_adaptation: bool,
        preferred_source: TextSource | None,
        extraction_id: UUID | None,
    ) -> AdaptationResponse:
        # Store original / variants / selected as separate files.
        base = self.adaptation_dir(adaptation_id)
        (base / "original.json").write_text(
            json.dumps({"original_text": original_text}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (base / "variants.json").write_text(
            json.dumps(
                [v.model_dump(mode="json") for v in variants],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (base / "selected.json").write_text(
            json.dumps({"selected_text": selected_text}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        meta = {
            "adaptation_id": str(adaptation_id),
            "skip_adaptation": skip_adaptation,
            "preferred_source": preferred_source,
            "extraction_id": str(extraction_id) if extraction_id else None,
        }
        (base / "meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        response = AdaptationResponse(
            adaptation_id=adaptation_id,
            original_text=original_text,
            variants=variants,
            selected_text=selected_text,
            skip_adaptation=skip_adaptation,
            preferred_source=preferred_source,
            extraction_id=extraction_id,
        )
        (base / "adaptation.json").write_text(
            json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Saved adaptation_id=%s variants=%s selected=%s",
            adaptation_id,
            len(variants),
            selected_text is not None,
        )
        return response

    def load_adaptation(self, adaptation_id: UUID) -> AdaptationResponse:
        path = self.adaptation_dir(adaptation_id) / "adaptation.json"
        if not path.is_file():
            raise SessionNotFoundError(
                "Adaptation session not found",
                details={"adaptation_id": str(adaptation_id)},
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return AdaptationResponse.model_validate(data)

    def update_selected(self, adaptation_id: UUID, selected_text: str) -> AdaptationResponse:
        current = self.load_adaptation(adaptation_id)
        return self.save_adaptation(
            adaptation_id=adaptation_id,
            original_text=current.original_text,
            variants=current.variants,
            selected_text=selected_text,
            skip_adaptation=current.skip_adaptation,
            preferred_source=current.preferred_source,
            extraction_id=current.extraction_id,
        )


def compose_original_from_extraction(
    extraction: ExtractionResponse,
    *,
    preferred_source: TextSource | None = None,
) -> tuple[str, TextSource]:
    """Pick a default original text while keeping sources separate in storage."""
    on_screen = _join_fragments([f.text for f in extraction.on_screen_text])
    spoken = _join_fragments([f.text for f in extraction.spoken_text])

    if preferred_source == "on_screen" and on_screen:
        return on_screen, "on_screen"
    if preferred_source == "spoken" and spoken:
        return spoken, "spoken"
    if preferred_source == "composed" and (on_screen or spoken):
        parts = []
        if on_screen:
            parts.append(f"[on_screen] {on_screen}")
        if spoken:
            parts.append(f"[spoken] {spoken}")
        return " | ".join(parts), "composed"

    # Default: prefer on-screen hook text, else spoken — never silently merge.
    if on_screen:
        return on_screen, "on_screen"
    if spoken:
        return spoken, "spoken"
    return "", "composed"


def _join_fragments(parts: list[str]) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return " ".join(cleaned)


__all__ = [
    "HookTextSessionStore",
    "compose_original_from_extraction",
]
