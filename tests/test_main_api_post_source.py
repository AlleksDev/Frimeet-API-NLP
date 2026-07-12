from app.modules.posts.infrastructure.main_api_post_source import (
    MainApiPostsClient,
    post_change_to_record,
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
            "author_type": "user",
            "author_id": "user_1",
            "source_version": 7,
            "created_at": "2026-07-05T10:00:00Z",
        }
    )

    assert record is not None
    assert record.id == "post_123"
    assert "Plan de cafe" in record.document
    assert record.metadata["title"] == "Plan de cafe"
    assert record.is_active is True
    assert record.author_id == "user_1"
    assert record.source_version == 7
    assert record.published_at is not None
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


def test_post_change_maps_tombstone() -> None:
    change = post_change_to_record(
        {
            "event_id": 19,
            "post_id": "post_1",
            "event_type": "post.deleted",
            "source_version": 8,
        }
    )
    assert change is not None
    assert change.operation == "delete"
    assert change.post is None
