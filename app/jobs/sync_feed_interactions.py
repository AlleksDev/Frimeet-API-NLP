import asyncio

from app.modules.posts.application.use_cases.update_user_interest_profiles import UpdateUserInterestProfilesUseCase
from app.modules.posts.infrastructure.aws_pgvector_user_profile_repository import AwsPgvectorUserProfileRepository
from app.modules.posts.infrastructure.main_api_feed_interaction_source import MainApiFeedInteractionSource
from app.shared.config.settings import get_settings
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient


async def main() -> None:
    settings = get_settings()
    if settings.vector_store_provider != "aws_pgvector":
        raise RuntimeError("sync_feed_interactions requiere VECTOR_STORE_PROVIDER=aws_pgvector")
    use_case = UpdateUserInterestProfilesUseCase(
        MainApiFeedInteractionSource(settings),
        AwsPgvectorUserProfileRepository(
            AwsPgvectorClient(settings, role="writer")
        ),
        half_life_days=settings.user_profile_half_life_days,
    )
    last_event_id = await use_case.execute()
    print(f"last_interaction_event_id={last_event_id}")


if __name__ == "__main__":
    asyncio.run(main())
