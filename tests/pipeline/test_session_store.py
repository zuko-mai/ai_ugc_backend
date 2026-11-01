import pytest

from app.pipeline.store import SessionStore


@pytest.mark.asyncio
async def test_session_store_round_trip(storage):
    store = SessionStore(storage)
    created = await store.create()

    loaded = await store.get(created.session_id)

    assert loaded.session_id == created.session_id
    assert loaded.step == 0
    assert storage.get_local_path(
        f"sessions/{created.session_id}/session.json"
    ).is_file()


def test_create_session_set_product_and_enforce_order(client):
    created = client.post("/v1/pipeline/sessions")
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    product = client.put(
        f"/v1/pipeline/sessions/{session_id}/product",
        json={
            "name": "Widget",
            "description": "A compact widget",
            "target_audience": "creators",
            "language": "en",
        },
    )
    assert product.status_code == 200
    assert product.json()["product"]["description"] == "A compact widget"

    skipped = client.post(
        f"/v1/pipeline/sessions/{session_id}/hook/candidates",
        json={},
    )
    assert skipped.status_code == 409
    assert skipped.json()["error_code"] == "pipeline_order_error"
