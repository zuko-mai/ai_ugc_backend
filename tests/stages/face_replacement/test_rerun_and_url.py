from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings


def _write_jpeg(path: Path, *, width: int, height: int) -> None:
    img = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 3), min(width, height) // 6, (180, 160, 140), -1)
    assert cv2.imwrite(str(path), img)


@pytest.fixture
def frame_and_face(settings: Settings, tmp_path: Path) -> tuple[str, Path]:
    frame_path = tmp_path / "frame.jpg"
    face_path = tmp_path / "face.jpg"
    _write_jpeg(frame_path, width=640, height=640)
    _write_jpeg(face_path, width=320, height=320)
    dest = settings.storage_root / "test_inputs" / "frame_rerun.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(frame_path.read_bytes())
    frame_url = f"{settings.public_base_url}/media/test_inputs/frame_rerun.jpg"
    return frame_url, face_path


def _generate(client: TestClient, *, video_id: str, frame_url: str, face_path: Path, **extra):
    data = {
        "video_id": video_id,
        "selected_frame_url": frame_url,
        "model_provider": "stub",
        "model_name": "stub-passthrough",
        **{k: str(v) for k, v in extra.items() if v is not None},
    }
    with face_path.open("rb") as fh:
        return client.post(
            "/v1/face-replacement/generate",
            data=data,
            files={"face_reference": ("face.jpg", fh, "image/jpeg")},
        )


def test_face_rerun_reuses_archived_reference(
    client: TestClient, frame_and_face: tuple[str, Path]
) -> None:
    frame_url, face_path = frame_and_face
    video_id = str(uuid4())
    first = _generate(
        client,
        video_id=video_id,
        frame_url=frame_url,
        face_path=face_path,
        user_prompt="keep natural skin texture",
    )
    assert first.status_code == 200, first.text
    generation_id = first.json()["generation_id"]

    rerun = client.post(
        f"/v1/face-replacement/generations/{generation_id}/rerun",
        json={"user_prompt": "slightly warmer light"},
    )
    assert rerun.status_code == 200, rerun.text
    assert rerun.json()["generation_id"] != generation_id

    stored = client.get(f"/v1/face-replacement/generations/{generation_id}/request")
    assert stored.status_code == 200
    assert stored.json()["user_prompt"] == "keep natural skin texture"
    assert stored.json()["face_reference_key"]


def test_face_generate_from_reference_url(
    client: TestClient, frame_and_face: tuple[str, Path]
) -> None:
    frame_url, face_path = frame_and_face
    first = _generate(
        client, video_id=str(uuid4()), frame_url=frame_url, face_path=face_path
    )
    assert first.status_code == 200, first.text
    meta = client.get(
        f"/v1/face-replacement/generations/{first.json()['generation_id']}/request"
    ).json()
    face_url = meta["face_reference_url"]
    assert face_url

    second = client.post(
        "/v1/face-replacement/generate",
        data={
            "video_id": str(uuid4()),
            "selected_frame_url": frame_url,
            "face_reference_url": face_url,
            "model_provider": "stub",
            "model_name": "stub-passthrough",
        },
    )
    assert second.status_code == 200, second.text
