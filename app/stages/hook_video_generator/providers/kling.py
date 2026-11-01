"""Kling 3.0 image-to-video adapter.

Maps our provider-agnostic request onto Kling's async create + poll API.
Uses the prepaid General pack (not Kling 4.0).
"""

from __future__ import annotations

import base64
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from app.core.logging import get_logger
from app.stages.hook_video_generator.exceptions import (
    ProviderApiError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
)
from app.stages.hook_video_generator.providers.base import (
    VideoGenerationProvider,
    VideoProviderPoll,
    VideoProviderRequest,
    VideoProviderSubmission,
    normalize_remote_status,
)

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api-singapore.klingai.com"
DEFAULT_MODEL = "kling-3.0"
PROMPT_LIMIT = 3072
MIN_KLING_DURATION = 3
MAX_KLING_HOOK_DURATION = 4

_MODEL_PATHS: dict[str, str] = {
    "kling-3.0": "/image-to-video/kling-3.0",
    "kling-3": "/image-to-video/kling-3.0",
    "kling3": "/image-to-video/kling-3.0",
    "kling-3.0-omni": "/image-to-video/kling-3.0",
    "kling-3.0-turbo": "/image-to-video/kling-3.0-turbo",
    "kling-3-turbo": "/image-to-video/kling-3.0-turbo",
    "kling3-turbo": "/image-to-video/kling-3.0-turbo",
}


def resolve_kling_model(model_name: str) -> tuple[str, str]:
    """Return (canonical model id, create path)."""
    key = model_name.strip().lower()
    path = _MODEL_PATHS.get(key)
    if path is None:
        raise ProviderApiError(
            f"Unsupported Kling model '{model_name}'",
            details={
                "model_name": model_name,
                "supported": sorted(set(_MODEL_PATHS)),
            },
        )
    canonical = "kling-3.0-turbo" if "turbo" in key else "kling-3.0"
    return canonical, path


def map_kling_duration(duration_seconds: float) -> tuple[int, list[str]]:
    """Kling 3.0 image-to-video accepts whole seconds starting at 3."""
    warnings: list[str] = []
    if duration_seconds < MIN_KLING_DURATION:
        raise ProviderApiError(
            "Kling 3.0 requires duration_seconds of 3 or 4 for this stage",
            details={
                "duration_seconds": duration_seconds,
                "allowed": [MIN_KLING_DURATION, MAX_KLING_HOOK_DURATION],
            },
        )
    rounded = int(round(duration_seconds))
    if rounded < MIN_KLING_DURATION:
        rounded = MIN_KLING_DURATION
    if rounded > MAX_KLING_HOOK_DURATION:
        rounded = MAX_KLING_HOOK_DURATION
    if abs(duration_seconds - rounded) > 0.01:
        warnings.append(
            f"kling_duration_rounded: requested {duration_seconds:.2f}s → {rounded}s"
        )
    return rounded, warnings


def map_kling_resolution(width: int, height: int) -> str:
    short_edge = min(width, height)
    long_edge = max(width, height)
    if short_edge >= 2160 or long_edge >= 3840:
        return "4k"
    if short_edge >= 1080:
        return "1080p"
    return "720p"


def encode_frame_payload(request: VideoProviderRequest) -> str:
    """Prefer a public HTTPS URL; otherwise send the frame as a data URL."""
    source = request.first_frame.source_url.strip()
    if _is_public_http_url(source):
        return source
    mime = request.first_frame.mime_type or "image/jpeg"
    if mime not in {"image/jpeg", "image/jpg", "image/png"}:
        mime = "image/jpeg"
    encoded = base64.b64encode(request.first_frame.data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_kling_prompt(request: VideoProviderRequest) -> str:
    prompt = request.instructions.prompt.strip()
    negative = request.instructions.negative_prompt.strip()
    if negative:
        suffix = f" Avoid: {negative}."
        if len(prompt) + len(suffix) <= PROMPT_LIMIT:
            prompt = f"{prompt}{suffix}"
    if len(prompt) > PROMPT_LIMIT:
        prompt = prompt[: PROMPT_LIMIT - 1].rstrip() + "…"
    return prompt


class KlingVideoProvider(VideoGenerationProvider):
    name = "kling"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 300,
        audio: str = "off",
        multi_shot: bool = False,
        prefer_base64: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ProviderNotConfiguredError(
                "Kling API key is not configured",
                details={"required_settings": ["kling_api_key"]},
            )
        if audio not in {"off", "native"}:
            raise ProviderNotConfiguredError(
                "kling_audio must be 'off' or 'native'",
                details={"kling_audio": audio},
            )
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.audio = audio
        self.multi_shot = multi_shot
        self.prefer_base64 = prefer_base64
        self._client = client

    async def submit(self, request: VideoProviderRequest) -> VideoProviderSubmission:
        canonical, path = resolve_kling_model(request.model_name or DEFAULT_MODEL)
        duration, warnings = map_kling_duration(request.duration_seconds)
        resolution = map_kling_resolution(request.width, request.height)
        frame = (
            _data_url(request)
            if self.prefer_base64
            else encode_frame_payload(request)
        )
        if frame.startswith("data:"):
            warnings.append("kling_first_frame_base64: local/private URL was inlined")
        payload = {
            "contents": [
                {"type": "prompt", "text": build_kling_prompt(request)},
                {"type": "first_frame", "url": frame},
            ],
            "settings": {
                "resolution": resolution,
                "duration": duration,
                "audio": self.audio,
                "multi_shot": self.multi_shot,
            },
            "options": {
                "watermark_info": {"enabled": False},
            },
        }
        body = await self._request("POST", path, json_body=payload)
        data = _require_object(body.get("data"), field="data")
        provider_job_id = str(data.get("id") or "").strip()
        if not provider_job_id:
            raise ProviderApiError(
                "Kling create response did not include a task id",
                details={"body": _safe_body(body)},
            )
        raw_status = str(data.get("status") or "submitted")
        status = _map_open_status(raw_status)
        logger.info(
            "Kling task submitted id=%s model=%s status=%s duration=%s resolution=%s",
            provider_job_id,
            canonical,
            status,
            duration,
            resolution,
        )
        return VideoProviderSubmission(
            provider_job_id=provider_job_id,
            status=status,
            provider_warnings=tuple(warnings),
        )

    async def poll(self, provider_job_id: str) -> VideoProviderPoll:
        path = f"/tasks?task_ids={quote(provider_job_id, safe='')}"
        body = await self._request("GET", path, json_body=None)
        items = body.get("data")
        if not isinstance(items, list) or not items:
            raise ProviderApiError(
                "Kling task was not found",
                details={"provider_job_id": provider_job_id},
            )
        task = _pick_task(items, provider_job_id)
        raw_status = str(task.get("status") or "").strip()
        if not raw_status:
            raise ProviderApiError(
                "Kling poll response did not include a status",
                details={"provider_job_id": provider_job_id},
            )
        try:
            status = normalize_remote_status(raw_status)
        except ValueError as exc:
            raise ProviderApiError(
                f"Unrecognized Kling status '{raw_status}'",
                details={"status": raw_status, "provider_job_id": provider_job_id},
            ) from exc

        if status == "failed":
            return VideoProviderPoll(
                provider_job_id=provider_job_id,
                status="failed",
                error_message=_task_error(task) or "Kling task failed",
            )
        if status in {"queued", "processing"}:
            return VideoProviderPoll(
                provider_job_id=provider_job_id,
                status=status,
            )

        video_url = _video_url(task)
        if not video_url:
            return VideoProviderPoll(
                provider_job_id=provider_job_id,
                status="failed",
                error_message="Kling marked the task succeeded without a video URL",
            )
        return VideoProviderPoll(
            provider_job_id=provider_job_id,
            status="succeeded",
            video_url=video_url,
            mime_type="video/mp4",
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}" if path.startswith("/") else f"{self.base_url}/{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            if self._client is not None:
                response = await self._client.request(
                    method, url, json=json_body, headers=headers
                )
            else:
                timeout = httpx.Timeout(self.timeout_seconds)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.request(
                        method, url, json=json_body, headers=headers
                    )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "Kling API timed out",
                details={"url": url},
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderApiError(
                "Kling API request failed",
                details={"url": url, "error": str(exc)},
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderApiError(
                "Kling API returned a non-JSON response",
                details={"status_code": response.status_code, "body": response.text[:500]},
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderApiError("Kling API JSON response must be an object")

        code = payload.get("code")
        if response.status_code >= 400 or (isinstance(code, int) and code != 0):
            raise ProviderApiError(
                str(payload.get("message") or f"Kling API error code={code}"),
                details={
                    "status_code": response.status_code,
                    "code": code,
                    "body": _safe_body(payload),
                },
            )
        return payload


def _data_url(request: VideoProviderRequest) -> str:
    mime = request.first_frame.mime_type or "image/jpeg"
    if mime not in {"image/jpeg", "image/jpg", "image/png"}:
        mime = "image/jpeg"
    encoded = base64.b64encode(request.first_frame.data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _is_public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1", "testserver"}:
        return False
    if host.endswith(".local"):
        return False
    return True


def _map_open_status(raw: str) -> str:
    try:
        status = normalize_remote_status(raw)
    except ValueError as exc:
        raise ProviderApiError(
            f"Unrecognized Kling create status '{raw}'",
            details={"status": raw},
        ) from exc
    if status == "succeeded":
        raise ProviderApiError(
            "Kling reported completion on create. "
            "A video is accepted only after a later poll returns a file URL.",
            details={"status": raw},
        )
    if status == "failed":
        raise ProviderApiError(
            "Kling rejected the video task",
            details={"status": raw},
        )
    return status


def _require_object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProviderApiError(
            f"Kling response field '{field}' must be an object",
            details={"field": field},
        )
    return value


def _pick_task(items: list[Any], provider_job_id: str) -> dict[str, Any]:
    for item in items:
        if isinstance(item, dict) and str(item.get("id") or "") == provider_job_id:
            return item
    first = items[0]
    if isinstance(first, dict):
        return first
    raise ProviderApiError(
        "Kling poll response did not include a task object",
        details={"provider_job_id": provider_job_id},
    )


def _video_url(task: dict[str, Any]) -> str | None:
    outputs = task.get("outputs") or []
    if not isinstance(outputs, list):
        return None
    for item in outputs:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").lower() != "video":
            continue
        for key in ("url", "watermark_url"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _task_error(task: dict[str, Any]) -> str | None:
    for key in ("message", "error", "msg"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _safe_body(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop large base64 fields from error details."""
    clone = dict(payload)
    data = clone.get("data")
    if isinstance(data, dict):
        clone["data"] = {k: v for k, v in data.items() if k != "contents"}
    return clone
