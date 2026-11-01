from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.stages.hook_text_engine.schemas import (
    OnScreenTextFragment,
    SpokenTextFragment,
)


def _ingest_sample(client: TestClient, sample_mp4: Path) -> tuple[UUID, str]:
    with sample_mp4.open("rb") as fh:
        response = client.post(
            "/v1/ingestion/upload",
            files={"file": ("sample.mp4", fh, "video/mp4")},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    return UUID(body["video_id"]), body["video_url"]


@pytest.fixture
def mock_extractors(monkeypatch: pytest.MonkeyPatch):
    async def _fake_ocr(frames, **kwargs):  # noqa: ANN001, ANN003
        assert frames
        return [
            OnScreenTextFragment(
                text="WAIT FOR IT",
                start_seconds=0.0,
                end_seconds=0.5,
                bbox=[10, 20, 200, 40],
                confidence=0.91,
                language="en",
            )
        ]

    async def _fake_asr(audio_path, **kwargs):  # noqa: ANN001, ANN003
        if audio_path is None:
            return []
        return [
            SpokenTextFragment(
                text="you need to see this",
                start_seconds=0.2,
                end_seconds=1.0,
                language="en",
                confidence=0.88,
            )
        ]

    monkeypatch.setattr(
        "app.stages.hook_text_engine.service.extract_on_screen_text",
        _fake_ocr,
    )
    monkeypatch.setattr(
        "app.stages.hook_text_engine.service.extract_spoken_text",
        _fake_asr,
    )


def test_extract_keeps_on_screen_and_spoken_separate(
    client: TestClient, sample_mp4: Path, mock_extractors
) -> None:
    video_id, video_url = _ingest_sample(client, sample_mp4)

    response = client.post(
        "/v1/hook-text-engine/extract",
        json={
            "video_id": str(video_id),
            "video_url": video_url,
            "start_seconds": 0,
            "end_seconds": 1,
            "languages": ["en", "ru"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["video_id"] == str(video_id)
    assert body["extraction_id"]
    assert len(body["on_screen_text"]) == 1
    assert body["on_screen_text"][0]["text"] == "WAIT FOR IT"
    assert body["on_screen_text"][0]["source"] == "on_screen"
    assert body["on_screen_text"][0]["bbox"] == [10, 20, 200, 40]
    assert len(body["spoken_text"]) == 1
    assert body["spoken_text"][0]["text"] == "you need to see this"
    assert body["spoken_text"][0]["source"] == "spoken"
    assert body["spoken_text"][0]["language"] == "en"
    # Sources must not be silently merged into one field.
    assert "merged_text" not in body


def test_extract_without_audio_returns_empty_spoken(
    client: TestClient, sample_mp4_no_audio: Path, mock_extractors, monkeypatch
) -> None:
    async def _no_speech(audio_path, **kwargs):  # noqa: ANN001, ANN003
        assert audio_path is None
        return []

    monkeypatch.setattr(
        "app.stages.hook_text_engine.service.extract_spoken_text",
        _no_speech,
    )

    video_id, video_url = _ingest_sample(client, sample_mp4_no_audio)
    response = client.post(
        "/v1/hook-text-engine/extract",
        json={
            "video_id": str(video_id),
            "video_url": video_url,
            "start_seconds": 0,
            "end_seconds": 0.5,
            "languages": ["en"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["on_screen_text"]
    assert body["spoken_text"] == []


def test_adapt_and_select_variant(client: TestClient) -> None:
    adapt = client.post(
        "/v1/hook-text-engine/adapt",
        json={
            "original_text": "Wait… nobody talks about this?",
            "product_name": "AquaBottle",
            "product_description": "Insulated bottle that keeps drinks cold for hours.",
            "target_audience": "commuters",
            "language": "en",
            "hook_goal": "curiosity",
            "variants_count": 3,
        },
    )
    assert adapt.status_code == 200, adapt.text
    body = adapt.json()
    assert body["original_text"] == "Wait… nobody talks about this?"
    assert body["selected_text"] is None
    assert len(body["variants"]) == 3
    assert body["skip_adaptation"] is False

    variant_id = body["variants"][0]["variant_id"]
    select = client.post(
        "/v1/hook-text-engine/select",
        json={
            "adaptation_id": body["adaptation_id"],
            "variant_id": variant_id,
        },
    )
    assert select.status_code == 200, select.text
    selected = select.json()
    assert selected["selection_mode"] == "variant"
    assert selected["selected_text"] == body["variants"][0]["hook_text"]

    session = client.get(f"/v1/hook-text-engine/sessions/{body['adaptation_id']}")
    assert session.status_code == 200
    assert session.json()["selected_text"] == selected["selected_text"]
    assert session.json()["original_text"] == body["original_text"]
    assert len(session.json()["variants"]) == 3


def test_manual_text_priority_over_original(client: TestClient) -> None:
    adapt = client.post(
        "/v1/hook-text-engine/adapt",
        json={
            "original_text": "auto recognized hook",
            "manual_text": "My edited hook for AquaBottle",
            "product_name": "AquaBottle",
            "product_description": "Insulated bottle for cold drinks.",
            "target_audience": "athletes",
            "language": "en",
            "hook_goal": "benefit",
            "variants_count": 2,
        },
    )
    assert adapt.status_code == 200, adapt.text
    body = adapt.json()
    assert body["original_text"] == "My edited hook for AquaBottle"
    assert body["preferred_source"] == "manual"


def test_skip_adaptation_keeps_original_as_selected(client: TestClient) -> None:
    adapt = client.post(
        "/v1/hook-text-engine/adapt",
        json={
            "original_text": "Keep this exact hook",
            "product_name": "AquaBottle",
            "product_description": "Insulated bottle.",
            "target_audience": "commuters",
            "language": "en",
            "hook_goal": "curiosity",
            "variants_count": 5,
            "skip_adaptation": True,
        },
    )
    assert adapt.status_code == 200, adapt.text
    body = adapt.json()
    assert body["skip_adaptation"] is True
    assert body["variants"] == []
    assert body["selected_text"] == "Keep this exact hook"


def test_select_manual_overrides_variant(client: TestClient) -> None:
    adapt = client.post(
        "/v1/hook-text-engine/adapt",
        json={
            "original_text": "Original hook",
            "product_name": "FocusTea",
            "product_description": "Herbal tea for calm focus.",
            "target_audience": "students",
            "language": "en",
            "hook_goal": "question",
            "variants_count": 2,
        },
    )
    body = adapt.json()
    select = client.post(
        "/v1/hook-text-engine/select",
        json={
            "adaptation_id": body["adaptation_id"],
            "variant_id": body["variants"][0]["variant_id"],
            "manual_text": "Final user-edited hook",
        },
    )
    assert select.status_code == 200
    assert select.json()["selection_mode"] == "manual"
    assert select.json()["selected_text"] == "Final user-edited hook"


def test_select_use_original(client: TestClient) -> None:
    adapt = client.post(
        "/v1/hook-text-engine/adapt",
        json={
            "original_text": "Stay with original",
            "product_name": "FocusTea",
            "product_description": "Herbal tea.",
            "target_audience": "students",
            "language": "en",
            "hook_goal": "curiosity",
            "variants_count": 2,
        },
    )
    body = adapt.json()
    select = client.post(
        "/v1/hook-text-engine/select",
        json={
            "adaptation_id": body["adaptation_id"],
            "use_original": True,
        },
    )
    assert select.status_code == 200
    assert select.json()["selection_mode"] == "original"
    assert select.json()["selected_text"] == "Stay with original"


def test_adapt_linked_missing_extraction_404(client: TestClient) -> None:
    response = client.post(
        "/v1/hook-text-engine/adapt",
        json={
            "original_text": "Hook",
            "product_name": "X",
            "product_description": "Y",
            "target_audience": "Z",
            "language": "en",
            "extraction_id": str(uuid4()),
        },
    )
    assert response.status_code == 404
    assert response.json()["error_code"] == "session_not_found"


def test_invalid_interval_rejected(client: TestClient, sample_mp4: Path) -> None:
    video_id, video_url = _ingest_sample(client, sample_mp4)
    response = client.post(
        "/v1/hook-text-engine/extract",
        json={
            "video_id": str(video_id),
            "video_url": video_url,
            "start_seconds": 3,
            "end_seconds": 1,
            "languages": ["en"],
        },
    )
    assert response.status_code == 422


def test_extract_requires_video_ref(client: TestClient) -> None:
    response = client.post(
        "/v1/hook-text-engine/extract",
        json={
            "start_seconds": 0,
            "end_seconds": 2,
            "languages": ["en"],
        },
    )
    assert response.status_code == 422
