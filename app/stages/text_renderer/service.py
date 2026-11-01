"""Text Renderer stage — deterministic FFmpeg burn-in for UGC captions."""

from __future__ import annotations

import re
from uuid import UUID, uuid4

from app.core.config import Settings
from app.core.logging import get_logger
from app.stages.text_renderer.ass import prepare_segments
from app.stages.text_renderer.exceptions import (
    InvalidTextError,
    ProbeError,
    RenderNotFoundError,
)
from app.stages.text_renderer.fetch import media_key_from_url, read_video_bytes
from app.stages.text_renderer.probe import ProbedVideo, probe_bytes
from app.stages.text_renderer.render import (
    burn_text_video,
    export_without_text,
    render_preview_frame,
)
from app.stages.text_renderer.schemas import (
    StoredRenderRequest,
    TextPreviewRequest,
    TextPreviewResponse,
    TextRenderRequest,
    TextRenderResponse,
    TextRerenderRequest,
)
from app.stages.text_renderer.store import TextRendererStore
from app.storage.base import FileStorage

logger = get_logger(__name__)

_PREVIOUS_OUTPUT_KEY = re.compile(
    r"^text_renderer/renders/"
    r"(?P<render_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    r"/output\.(?:mp4|mov|m4v)$"
)


class TextRendererService:
    def __init__(self, storage: FileStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self.store = TextRendererStore(storage)

    async def render(self, request: TextRenderRequest) -> TextRenderResponse:
        source_bytes, mime, probed, source_key, load_warnings = await self._load_and_archive(
            request
        )
        return await self._render_from_source(
            request=request,
            source_bytes=source_bytes,
            mime=mime,
            probed=probed,
            source_key=source_key,
            parent_render_id=None,
            include_preview=True,
            extra_warnings=load_warnings,
        )

    async def preview(self, request: TextPreviewRequest) -> TextPreviewResponse:
        source_bytes, _mime, probed, _source_key, load_warnings = await self._load_and_archive(
            request
        )
        warnings: list[str] = list(load_warnings)
        if not probed.has_video or probed.width < 2 or probed.height < 2:
            raise ProbeError(
                "Source file has no usable video stream",
                details={"width": probed.width, "height": probed.height},
            )

        if request.skip_text:
            # Still frame without overlay.
            prepared = []
            segments_count = 0
        else:
            prepared = prepare_segments(
                text=request.text,
                segments=request.segments,
                style=request.style,
                frame_width=probed.width,
                duration_seconds=probed.duration_seconds,
            )
            segments_count = len(prepared)
            if segments_count == 0:
                warnings.append("no_segments_in_duration")

        at = request.preview_at_seconds
        if at is None:
            at = prepared[0].start_seconds if prepared else 0.0
        if probed.duration_seconds > 0:
            at = min(at, max(0.0, probed.duration_seconds - 0.05))

        suffix = _suffix_from_url(request.generated_video_url, None)
        jpeg = await render_preview_frame(
            source_bytes,
            style=request.style,
            segments=[] if request.skip_text else prepared,
            width=probed.width,
            height=probed.height,
            preview_at_seconds=at,
            source_suffix=suffix,
        )

        preview_key = await self.store.save_standalone_preview(request.video_id, jpeg)
        return TextPreviewResponse(
            video_id=request.video_id,
            preview_image_url=self.storage.get_url(preview_key),
            width=probed.width,
            height=probed.height,
            preview_at_seconds=round(at, 3),
            text_segments_rendered=segments_count,
            warnings=warnings,
        )

    async def rerender(
        self,
        render_id: UUID,
        body: TextRerenderRequest,
    ) -> TextRenderResponse:
        stored = await self.store.get_request(render_id)
        record = await self.store.get_record(render_id)

        style = body.style or stored.style
        text = stored.text if body.text is None else body.text
        segments = stored.segments if body.segments is None else body.segments
        language = body.language or stored.language
        output_format = body.output_format or stored.output_format
        skip_text = stored.skip_text if body.skip_text is None else body.skip_text

        if not skip_text and not segments and not text:
            raise InvalidTextError(
                "rerender requires text/segments or skip_text=true",
                details={"render_id": str(render_id)},
            )

        request = TextRenderRequest(
            video_id=stored.video_id,
            generated_video_url=stored.generated_video_url,
            text=text,
            language=language,
            style=style,
            segments=segments,
            output_format=output_format,
            skip_text=skip_text,
        )

        path = self.storage.get_local_path(stored.source_video_key)
        source_bytes = path.read_bytes()
        probed = await probe_bytes(source_bytes, suffix=path.suffix or ".mp4")
        return await self._render_from_source(
            request=request,
            source_bytes=source_bytes,
            mime=None,
            probed=probed,
            source_key=stored.source_video_key,
            parent_render_id=record.render_id,
            include_preview=True,
            extra_warnings=[],
        )

    async def get_render(self, render_id: UUID) -> TextRenderResponse:
        record = await self.store.get_record(render_id)
        return TextRenderResponse(
            video_id=record.video_id,
            render_id=record.render_id,
            asset_id=record.render_id,
            rendered_video_url=self.storage.get_url(record.output_key),
            duration_seconds=record.duration_seconds,
            width=record.width,
            height=record.height,
            format=record.format,
            text_segments_rendered=record.text_segments_rendered,
            status="success",
            preview_image_url=(
                self.storage.get_url(record.preview_key) if record.preview_key else None
            ),
            warnings=record.warnings,
        )

    async def get_request(self, render_id: UUID) -> StoredRenderRequest:
        return await self.store.get_request(render_id)

    async def _load_and_archive(
        self,
        request: TextRenderRequest,
    ) -> tuple[bytes, str | None, ProbedVideo, str, list[str]]:
        """Load bytes to burn onto.

        If `generated_video_url` points at a previous text-renderer output, reuse that
        render's archived clean source — otherwise old burned text stays in the pixels
        and stacks under the new caption.
        """
        warnings: list[str] = []
        max_bytes = self.settings.max_text_render_video_size_mb * 1024 * 1024

        clean = await self._resolve_clean_source_from_previous_output(
            request.generated_video_url
        )
        if clean is not None:
            data, mime, source_key, note = clean
            warnings.append(note)
        else:
            data, mime = await read_video_bytes(
                request.generated_video_url,
                settings=self.settings,
                storage=self.storage,
                max_bytes=max_bytes,
            )
            suffix = _suffix_from_url(request.generated_video_url, mime)
            content_type = mime or ("video/quicktime" if suffix == ".mov" else "video/mp4")
            source_key = await self.store.archive_source(
                request.video_id,
                data,
                suffix=suffix,
                content_type=content_type,
            )

        suffix = _suffix_from_url(request.generated_video_url, mime)
        if source_key.endswith(".mov"):
            suffix = ".mov"
        elif source_key.endswith(".m4v"):
            suffix = ".m4v"
        elif source_key.endswith(".mp4"):
            suffix = ".mp4"

        probed = await probe_bytes(data, suffix=suffix)
        if not probed.has_video or probed.width < 2 or probed.height < 2:
            raise ProbeError(
                "Source file has no usable video stream",
                details={"width": probed.width, "height": probed.height},
            )
        return data, mime, probed, source_key, warnings

    async def _resolve_clean_source_from_previous_output(
        self,
        url: str,
    ) -> tuple[bytes, str | None, str, str] | None:
        key = media_key_from_url(url, self.settings.public_base_url)
        if not key:
            return None
        match = _PREVIOUS_OUTPUT_KEY.match(key.replace("\\", "/").lstrip("/"))
        if not match:
            return None
        previous_id = UUID(match.group("render_id"))
        try:
            stored = await self.store.get_request(previous_id)
        except RenderNotFoundError:
            return None
        source_key = stored.source_video_key
        if not await self.storage.exists(source_key):
            return None
        path = self.storage.get_local_path(source_key)
        data = path.read_bytes()
        mime = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".m4v": "video/mp4",
        }.get(path.suffix.lower())
        note = (
            "reused_clean_source_from_previous_render:"
            f"{previous_id} (avoided stacking text on burned output)"
        )
        logger.info(
            "Resolved previous text-renderer output to clean source previous=%s key=%s",
            previous_id,
            source_key,
        )
        return data, mime, source_key, note

    async def _render_from_source(
        self,
        *,
        request: TextRenderRequest,
        source_bytes: bytes,
        mime: str | None,
        probed: ProbedVideo,
        source_key: str,
        parent_render_id: UUID | None,
        include_preview: bool,
        extra_warnings: list[str] | None = None,
    ) -> TextRenderResponse:
        warnings: list[str] = list(extra_warnings or [])
        suffix = _suffix_from_url(request.generated_video_url, mime)
        render_id = uuid4()

        if request.skip_text:
            rendered = await export_without_text(
                source_bytes,
                output_format=request.output_format,
                source_suffix=suffix,
            )
            segments_count = 0
            preview_jpeg = None
        else:
            prepared = prepare_segments(
                text=request.text,
                segments=request.segments,
                style=request.style,
                frame_width=probed.width,
                duration_seconds=probed.duration_seconds,
            )
            segments_count = len(prepared)
            if segments_count == 0:
                warnings.append("no_segments_in_duration")
                rendered = await export_without_text(
                    source_bytes,
                    output_format=request.output_format,
                    source_suffix=suffix,
                )
                preview_jpeg = None
            else:
                preview_at = prepared[0].start_seconds if include_preview else None
                rendered = await burn_text_video(
                    source_bytes,
                    style=request.style,
                    segments=prepared,
                    width=probed.width,
                    height=probed.height,
                    output_format=request.output_format,
                    preview_at_seconds=preview_at,
                    source_suffix=suffix,
                )
                preview_jpeg = rendered.preview_jpeg

        # Confirm output geometry (vertical-friendly; do not force resize).
        out_probe = await probe_bytes(rendered.video_bytes, suffix=rendered.extension)
        if out_probe.width and out_probe.height:
            if (out_probe.width, out_probe.height) != (probed.width, probed.height):
                warnings.append(
                    f"output_dimensions_changed:"
                    f"{probed.width}x{probed.height}->{out_probe.width}x{out_probe.height}"
                )

        output_key = await self.store.save_output(
            render_id,
            rendered.video_bytes,
            extension=rendered.extension,
        )
        preview_key = None
        if preview_jpeg:
            preview_key = await self.store.save_preview(render_id, preview_jpeg)

        stored = StoredRenderRequest(
            video_id=request.video_id,
            generated_video_url=request.generated_video_url,
            source_video_key=source_key,
            text=request.text,
            language=request.language,
            style=request.style,
            segments=request.segments,
            output_format=request.output_format,
            skip_text=request.skip_text,
        )
        await self.store.save_request(render_id, stored)

        duration = out_probe.duration_seconds or probed.duration_seconds
        width = out_probe.width or probed.width
        height = out_probe.height or probed.height
        record = self.store.new_record(
            render_id=render_id,
            video_id=request.video_id,
            source_video_key=source_key,
            output_key=output_key,
            preview_key=preview_key,
            duration_seconds=round(duration, 3),
            width=width,
            height=height,
            output_format=request.output_format,
            text_segments_rendered=segments_count,
            warnings=warnings,
            parent_render_id=parent_render_id,
        )
        await self.store.save_record(record)

        logger.info(
            "Text render done video_id=%s render_id=%s segments=%s skip=%s %sx%s",
            request.video_id,
            render_id,
            segments_count,
            request.skip_text,
            width,
            height,
        )
        return TextRenderResponse(
            video_id=request.video_id,
            render_id=render_id,
            asset_id=render_id,
            rendered_video_url=self.storage.get_url(output_key),
            duration_seconds=record.duration_seconds,
            width=width,
            height=height,
            format=record.format,
            text_segments_rendered=segments_count,
            status="success",
            preview_image_url=(
                self.storage.get_url(preview_key) if preview_key else None
            ),
            warnings=warnings,
        )


def _suffix_from_url(url: str, mime: str | None) -> str:
    lower = url.lower().split("?", 1)[0]
    if lower.endswith(".mov"):
        return ".mov"
    if lower.endswith(".m4v"):
        return ".m4v"
    if mime and "quicktime" in mime:
        return ".mov"
    return ".mp4"

