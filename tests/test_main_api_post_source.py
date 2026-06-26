from app.modules.posts.infrastructure.main_api_post_source import (
    MainApiPostsClient,
    post_to_source_record,
)


def test_post_to_source_record_maps_api_post() -> None:
    record = post_to_source_record(
        {
            "id": "post_123",
            "title": "Plan de cafe",
            "city": "Tuxtla Gutierrez",
            "state": "Chiapas",
            "tags": ["cafe", "amigos"],
            "content": "Una publicacion para salir por cafe.",
            "is_active": True,
        }
    )

    assert record is not None
    assert record.id == "post_123"
    assert "Plan de cafe" in record.document
    assert record.metadata["title"] == "Plan de cafe"
    assert record.is_active is True
    assert len(record.content_hash) == 64


def test_extract_posts_supports_common_response_shapes() -> None:
    payload = {
        "data": {
            "items": [
                {"id": "post_1", "title": "Cafe"},
                {"id": "post_2", "title": "Parque"},
            ]
        }
    }

    posts = MainApiPostsClient._extract_posts(payload)

    assert [post["id"] for post in posts] == ["post_1", "post_2"]


def test_post_source_extracts_cursor_metadata() -> None:
    payload = {
        "data": [],
        "next_cursor": "cursor_posts",
        "has_more": True,
    }

    assert MainApiPostsClient._extract_next_cursor(payload) == "cursor_posts"
    assert MainApiPostsClient._extract_has_more(payload) is True
