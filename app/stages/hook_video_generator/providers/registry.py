from __future__ import annotations

from collections.abc import Callable

from app.core.config import Settings
from app.stages.hook_video_generator.exceptions import ProviderNotConfiguredError
from app.stages.hook_video_generator.providers.base import VideoGenerationProvider
from app.stages.hook_video_generator.providers.http import HttpVideoProvider
from app.stages.hook_video_generator.providers.kling import KlingVideoProvider
from app.stages.hook_video_generator.providers.stub import StubVideoProvider

ProviderFactory = Callable[[Settings], VideoGenerationProvider]


def _stub_factory(_settings: Settings) -> VideoGenerationProvider:
    return StubVideoProvider()


def _http_factory(settings: Settings) -> VideoGenerationProvider:
    submit_url = settings.hook_video_http_submit_url.strip()
    status_url = settings.hook_video_http_status_url.strip()
    if not submit_url or not status_url:
        raise ProviderNotConfiguredError(
            "HTTP video provider is not configured",
            details={
                "provider": "http",
                "required_settings": [
                    "hook_video_http_submit_url",
                    "hook_video_http_status_url",
                ],
            },
        )
    return HttpVideoProvider(
        submit_url=submit_url,
        status_url_template=status_url,
        api_key=settings.hook_video_http_api_key,
        timeout_seconds=float(settings.hook_video_timeout_seconds),
        auth_header=settings.hook_video_http_auth_header,
        auth_scheme=settings.hook_video_http_auth_scheme,
    )


def _kling_factory(settings: Settings) -> VideoGenerationProvider:
    return KlingVideoProvider(
        api_key=settings.kling_api_key,
        base_url=settings.kling_api_base_url,
        timeout_seconds=float(settings.hook_video_timeout_seconds),
        audio=settings.kling_audio,
        multi_shot=settings.kling_multi_shot,
        prefer_base64=settings.kling_prefer_base64,
    )


_REGISTRY: dict[str, ProviderFactory] = {
    "stub": _stub_factory,
    "http": _http_factory,
    "kling": _kling_factory,
}


def list_providers() -> list[str]:
    return sorted(_REGISTRY)


def get_provider(name: str, settings: Settings) -> VideoGenerationProvider:
    key = name.strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise ProviderNotConfiguredError(
            f"Unknown hook video provider '{name}'",
            details={"provider": name, "available": list_providers()},
        )
    return factory(settings)


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register an adapter. The public API keeps using model_provider and model_name."""
    _REGISTRY[name.strip().lower()] = factory


def unregister_provider(name: str) -> None:
    _REGISTRY.pop(name.strip().lower(), None)
