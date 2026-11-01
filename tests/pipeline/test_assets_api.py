def test_upload_and_get_asset(client):
    uploaded = client.post(
        "/v1/assets/upload",
        data={"kind": "face_reference"},
        files={"file": ("face.png", b"\x89PNG\r\nasset", "image/png")},
    )

    assert uploaded.status_code == 200
    body = uploaded.json()
    assert body["kind"] == "face_reference"
    assert body["filename"] == "face.png"
    assert body["size_bytes"] == 11
    assert body["url"].endswith(f"/media/uploads/{body['asset_id']}/file.png")

    fetched = client.get(f"/v1/assets/{body['asset_id']}")
    assert fetched.status_code == 200
    assert fetched.json() == body
