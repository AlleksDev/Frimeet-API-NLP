import pytest

from app.modules.clubs.infrastructure.main_api_club_source import MainApiClubsClient
from app.modules.events.infrastructure.main_api_event_source import MainApiEventsClient
from app.modules.groups.infrastructure.main_api_group_source import MainApiGroupsClient
from app.modules.users.infrastructure.main_api_user_source import MainApiUsersClient
from app.shared.config.settings import Settings


def test_search_sources_use_internal_snapshot_paths() -> None:
    settings = Settings(_env_file=None, ENV="local", MAIN_API_INTERNAL_TOKEN="internal")

    assert MainApiUsersClient(settings)._path == "api/v1/internal/search/users/snapshot"
    assert MainApiClubsClient(settings)._path == "api/v1/internal/search/clubs/snapshot"
    assert MainApiGroupsClient(settings)._path == "api/v1/internal/search/groups/snapshot"
    assert MainApiEventsClient(settings)._path == "api/v1/internal/search/events/snapshot"


def test_search_source_uses_only_internal_service_token() -> None:
    settings = Settings(
        _env_file=None,
        ENV="local",
        MAIN_API_INTERNAL_TOKEN="service-token",
        MAIN_API_AUTH_TOKEN="user-jwt-must-not-be-used",
    )

    assert MainApiUsersClient(settings)._build_headers() == {
        "Authorization": "Bearer service-token"
    }


def test_search_source_rejects_missing_internal_token_even_with_user_jwt() -> None:
    settings = Settings(
        _env_file=None,
        ENV="local",
        MAIN_API_AUTH_TOKEN="user-jwt-must-not-be-used",
    )

    with pytest.raises(RuntimeError, match="MAIN_API_INTERNAL_TOKEN"):
        MainApiUsersClient(settings)._build_headers()


def test_search_source_rejects_public_search_path() -> None:
    settings = Settings(
        _env_file=None,
        ENV="local",
        MAIN_API_INTERNAL_TOKEN="service-token",
        MAIN_API_USERS_SNAPSHOT_PATH="/api/v1/users/search",
    )

    with pytest.raises(ValueError, match="internal snapshot"):
        MainApiUsersClient(settings)
