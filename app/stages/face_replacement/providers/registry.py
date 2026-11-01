from __future__ import annotations

from collections.abc import Callable

from app.core.config import Settings
from app.stages.face_replacement.exceptions import ProviderNotConfiguredError
from app.stages.face_replacement.providers.base import FaceReplacementProvider
from app.stages.face_replacement.providers.gemini import GeminiFaceReplacementProvider
from app.stages.face_replacement.providers.stub import StubFaceReplacementProvider

ProviderFactory = Callable[[Settings], FaceReplacementProvider]


def _gemini_factory(settings: Settings) -> FaceReplacementProvider:
    return GeminiFaceReplacementProvider(api_key=settings.google_ai_api_key)


def _stub_factory(_settings: Settings) -> FaceReplacementProvider:
    return StubFaceReplacementProvider()


_REGISTRY: dict[str, ProviderFactory] = {
    "gemini": _gemini_factory,
    "stub": _stub_factory,
}


def list_providers() -> list[str]:
    return sorted(_REGISTRY)


def get_provider(name: str, settings: Settings) -> FaceReplacementProvider:
    key = name.strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise ProviderNotConfiguredError(
            f"Unknown face replacement provider '{name}'",
            details={"provider": name, "available": list_providers()},
        )
    return factory(settings)


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Allow tests / extensions to inject additional adapters."""
    _REGISTRY[name.strip().lower()] = factory
