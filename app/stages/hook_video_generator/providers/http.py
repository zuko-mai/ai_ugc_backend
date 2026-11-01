"""Configurable HTTP adapter for async image-to-video vendors.

The stage speaks this JSON contract. A vendor that uses different fields needs
its own `VideoGenerationProvider` subclass; the service stays unchanged.

Submit `POST submit_url`:

```json
{
  "model": "vendor-model-id",
  "prompt": "...",
  "negative_prompt": "...",
  "duration_seconds": 3,
  "aspect_ratio": "9:16",
  "width": 1080,
  "height": 1920,
  "fps": 30,
  "image_base64": "...",
  "image_mime_type": "image/jpeg"
}
```

Submit response: `{"id": "task-id", "status": "queued"}`.
`job_id` and `task_id` are accepted aliases. Status `succeeded` on submit is rejected.

Poll `GET status_url` with `{provider_job_id}` replaced:

```json
{"status": "succeeded", "video_url": "https://..."}
```

`url` and `output_url` are accepted aliases. `video_base64` is accepted inline.
"""

from __future__ import annotations

import base64
import json
from urllib.parse import quote

import httpx

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


class HttpVideoProvider(VideoGenerationProvider):
    name = "http"

    def __init__(
        self,
        *,
        submit_url: str,
        status_url_template: str,
        api_key: str,
        timeout_seconds: float,
        auth_header: str = "Authorization",
        auth_scheme: str = "Bearer",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if "{provider_job_id}" not in status_url_template:
            raise ProviderNotConfiguredError(
                "HTTP status URL must include {provider_job_id}",
                details={"status_url": status_url_template},
            )
        self.submit_url = submit_url
        self.status_url_template = status_url_template
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.auth_header = auth_header or "Authorization"
        self.auth_scheme = auth_scheme
        self._client = client

    async def submit(self, request: VideoProviderRequest) -> VideoProviderSubmission:
        payload = {
            "model": request.model_name,
            "prompt": request.instructions.prompt,
            "negative_prompt": request.instructions.negative_prompt,
            "duration_seconds": request.duration_seconds,
            "aspect_ratio": request.aspect_ratio,
            "width": request.width,
            "height": request.height,
            "fps": request.fps,
            "image_base64": base64.b64encode(request.first_frame.data).decode("ascii"),
            "image_mime_type": request.first_frame.mime_type,
        }
        body = await self._send("POST", self.submit_url, json_body=payload)
        raw_status = str(body.get("status") or "queued")
        status = _mapped_status(raw_status)
        if status == "succeeded":
            raise ProviderApiError(
                "Provider reported completion on submit. "
                "A video is accepted only after a later poll returns a file.",
                details={"status": raw_status},
            )
        if status == "failed":
            raise ProviderApiError(
                _error_message(body) or "Provider rejected the video task",
                details={"status": raw_status},
            )
        provider_job_id = _extract_id(body)
        return VideoProviderSubmission(
            provider_job_id=provider_job_id,
            status=status,  # type: ignore[arg-type]
            provider_warnings=_warnings(body),
        )

    async def poll(self, provider_job_id: str) -> VideoProviderPoll:
        url = self.status_url_template.replace(
            "{provider_job_id}",
            quote(provider_job_id, safe=""),
        )
        body = await self._send("GET", url, json_body=None)
        raw_status = body.get("status")
        if not isinstance(raw_status, str) or not raw_status.strip():
            raise ProviderApiError(
                "Provider poll response did not include a status",
                details={"provider_job_id": provider_job_id},
            )
        status = _mapped_status(raw_status)
        video_bytes = _video_bytes(body)
        return VideoProviderPoll(
            provider_job_id=provider_job_id,
            status=status,
            video_bytes=video_bytes,
            video_url=_video_url(body),
            mime_type="video/mp4" if video_bytes or _video_url(body) else None,
            error_message=_error_message(body) if status == "failed" else None,
            provider_warnings=_warnings(body),
        )

    async def _send(self, method: str, url: str, *, json_body: dict | None) -> dict:
        try:
            if self._client is not None:
                response = await self._client.request(
                    method,
                    url,
                    json=json_body,
                    headers=self._headers(),
                )
            else:
                timeout = httpx.Timeout(self.timeout_seconds)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.request(
                        method,
                        url,
                        json=json_body,
                        headers=self._headers(),
                    )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "Video provider timed out",
                details={"url": url},
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderApiError(
                "Video provider request failed",
                details={"url": url, "error": str(exc)},
            ) from exc

        if response.status_code >= 400:
            raise ProviderApiError(
                f"Video provider returned HTTP {response.status_code}",
                details={"url": url, "status_code": response.status_code, "body": response.text[:500]},
            )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderApiError(
                "Video provider returned a non-JSON response",
                details={"status_code": response.status_code},
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderApiError("Video provider JSON response must be an object")
        return payload

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        if self.auth_scheme:
            return {self.auth_header: f"{self.auth_scheme} {self.api_key}"}
        return {self.auth_header: self.api_key}


def _mapped_status(raw: str):
    try:
        return normalize_remote_status(raw)
    except ValueError as exc:
        raise ProviderApiError(
            f"Unrecognized provider status '{raw}'",
            details={"status": raw},
        ) from exc


def _extract_id(payload: dict) -> str:
    for key in ("id", "job_id", "task_id", "provider_job_id"):
        value = payload.get(key)
        if value:
            return str(value)
    raise ProviderApiError("Provider submit response did not include a job id")


def _video_url(payload: dict) -> str | None:
    for key in ("video_url", "url", "output_url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _video_bytes(payload: dict) -> bytes | None:
    raw = payload.get("video_base64")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return base64.b64decode(raw)
    except ValueError as exc:
        raise ProviderApiError("Provider video_base64 is not valid base64") from exc


def _error_message(payload: dict) -> str | None:
    for key in ("error", "message", "detail"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            return json.dumps(value)[:500]
    return None


def _warnings(payload: dict) -> tuple[str, ...]:
    raw = payload.get("warnings") or []
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw if item)
