from app.modules.clubs.infrastructure.main_api_club_source import club_to_source_record
from app.modules.events.infrastructure.main_api_event_source import event_to_source_record
from app.modules.groups.infrastructure.main_api_group_source import group_to_source_record
from app.modules.users.infrastructure.main_api_user_source import user_to_source_record


def test_user_document_excludes_sensitive_profile_fields() -> None:
    record = user_to_source_record(
        {
            "id": "user-1",
            "username": "tester1",
            "full_name": "Usuario Tester Uno",
            "email": "tester1@example.com",
            "birth_date": "2001-05-20",
            "current_location": {"latitude": 16.7531, "longitude": -93.1156},
            "bio": "Esta es mi biografia en Frimeet",
        }
    )
    assert record is not None
    assert "tester1" in record.document
    assert "biografia" in record.document
    assert "example.com" not in record.document
    assert "email" not in record.metadata
    assert "birth_date" not in record.metadata
    assert "current_location" not in record.metadata


def test_club_document_owns_club_fields_and_privacy_flag() -> None:
    record = club_to_source_record(
        {
            "id": "club-1",
            "name": "Club de Ajedrez Universitario",
            "description": "Partidas y entrenamiento dos veces por semana",
            "category": "ajedrez",
            "is_online": False,
            "is_private": False,
            "meeting_schedules": [{"day_of_week": 1, "start_time": "16:00"}],
        }
    )
    assert record is not None
    assert record.document.count("ajedrez") >= 5
    assert record.metadata["is_private"] is False
    assert record.metadata["meeting_schedules"][0]["day_of_week"] == 1


def test_event_identifier_tags_are_not_embedded_as_semantics() -> None:
    record = event_to_source_record(
        {
            "id": "event-1",
            "title": "Fiesta de Cumpleanos",
            "description": "Evento totalmente independiente",
            "is_public": True,
            "tags": ["7", "b83ab97e-91a4-4f69-b102-b27c6092e9cb"],
        }
    )
    assert record is not None
    assert "b83ab97e" not in record.document
    assert record.metadata["tag_ids"] == ["7", "b83ab97e-91a4-4f69-b102-b27c6092e9cb"]


def test_group_builds_private_search_acl_from_members_and_invites() -> None:
    record = group_to_source_record(
        {
            "id": "group-1",
            "creator_id": "creator",
            "name": "Amigos de la Uni",
            "description": "Grupo privado para organizar salidas",
            "is_private": True,
            "added_ids": ["friend"],
            "invited_ids": ["invitee"],
        }
    )
    assert record is not None
    assert record.metadata["authorized_user_ids"] == ["creator", "friend", "invitee"]
