"""Immutable input archival + generation history for face replacement."""

from __future__ import annotations

import hashlib
import json
from uuid import UUID, uuid4

from app.core.logging import get_logger
from app.stages.face_replacement.exceptions import GenerationNotFoundError
from app.stages.face_replacement.images import LoadedImage
from app.stages.face_replacement.schemas import GenerationRecord
from app.storage.base import FileStorage

logger = get_logger(__name__)


class FaceReplacementStore:
    def __init__(self, storage: FileStorage) -> None:
        self.storage = storage

    def inputs_prefix(self, video_id: UUID) -> str:
        return f"face_replacement/{video_id}/inputs"

    def generations_prefix(self, video_id: UUID) -> str:
        return f"face_replacement/{video_id}/generations"

    async def archive_input(self, video_id: UUID, role: str, image: LoadedImage) -> str:
        """Save an input image once (content-addressed). Never overwrite different content."""
        digest = hashlib.sha256(image.data).hexdigest()[:16]
        ext = _ext_for_mime(image.mime_type)
        key = f"{self.inputs_prefix(video_id)}/{role}_{digest}{ext}"
        if await self.storage.exists(key):
            logger.info("Input already archived key=%s role=%s", key, role)
            return key

        await self._save_bytes(key, image.data, content_type=image.mime_type)
        meta_key = f"{key}.meta.json"
        if not await self.storage.exists(meta_key):
            meta = {
                "role": role,
                "source_url": image.source_url,
                "width": image.width,
                "height": image.height,
                "mime_type": image.mime_type,
                "sha256_16": digest,
            }
            await self._save_bytes(
                meta_key,
                json.dumps(meta, indent=2).encode("utf-8"),
                content_type="application/json",
            )
        return key

    async def save_generation(
        self,
        *,
        video_id: UUID,
        image_bytes: bytes,
        mime_type: str,
        record: GenerationRecord,
    ) -> tuple[str, UUID]:
        generation_id = record.generation_id or uuid4()
        ext = _ext_for_mime(mime_type)
        image_key = f"{self.generations_prefix(video_id)}/{generation_id}/frame{ext}"
        meta_key = f"{self.generations_prefix(video_id)}/{generation_id}/meta.json"

        await self._save_bytes(image_key, image_bytes, content_type=mime_type)
        payload = record.model_copy(
            update={"generation_id": generation_id, "generated_frame_key": image_key}
        )
        await self._save_bytes(
            meta_key,
            payload.model_dump_json(indent=2).encode("utf-8"),
            content_type="application/json",
        )
        await self._save_bytes(
            self._index_key(generation_id),
            json.dumps(
                {
                    "generation_id": str(generation_id),
                    "video_id": str(video_id),
                    "meta_key": meta_key,
                    "generated_frame_key": image_key,
                },
                indent=2,
            ).encode("utf-8"),
            content_type="application/json",
        )
        logger.info(
            "Saved face replacement generation video_id=%s generation_id=%s key=%s",
            video_id,
            generation_id,
            image_key,
        )
        return image_key, generation_id

    async def get_generation(self, generation_id: UUID) -> GenerationRecord:
        index_key = self._index_key(generation_id)
        if not await self.storage.exists(index_key):
            raise GenerationNotFoundError(
                "Face replacement generation was not found",
                details={"generation_id": str(generation_id)},
            )
        index = json.loads(
            self.storage.get_local_path(index_key).read_text(encoding="utf-8")
        )
        meta_key = index["meta_key"]
        if not await self.storage.exists(meta_key):
            raise GenerationNotFoundError(
                "Face replacement generation metadata is missing",
                details={"generation_id": str(generation_id), "meta_key": meta_key},
            )
        return GenerationRecord.model_validate_json(
            self.storage.get_local_path(meta_key).read_text(encoding="utf-8")
        )

    async def _save_bytes(self, key: str, data: bytes, *, content_type: str) -> None:
        async def _stream():
            yield data

        await self.storage.save_stream(key, _stream(), content_type=content_type)

    @staticmethod
    def _index_key(generation_id: UUID) -> str:
        return f"face_replacement/index/{generation_id}.json"


def _ext_for_mime(mime_type: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }
    return mapping.get(mime_type.lower(), ".png")
