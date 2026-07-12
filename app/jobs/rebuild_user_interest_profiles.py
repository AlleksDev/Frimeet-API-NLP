import argparse
import asyncio

from app.modules.posts.application.use_cases.rebuild_user_interest_profiles import (
    RebuildUserInterestProfilesUseCase,
)
from app.modules.posts.application.use_cases.update_user_interest_profiles import (
    UpdateUserInterestProfilesUseCase,
)
from app.modules.posts.infrastructure.aws_pgvector_user_profile_repository import (
    AwsPgvectorUserProfileRepository,
)
from app.modules.posts.infrastructure.main_api_feed_interaction_source import (
    MainApiFeedInteractionSource,
)
from app.shared.config.settings import get_settings
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient


async def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruye perfiles semanticos")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--page-limit", type=int, default=1000)
    args = parser.parse_args()
    settings = get_settings()
    if settings.vector_store_provider != "aws_pgvector":
        raise RuntimeError("rebuild requiere VECTOR_STORE_PROVIDER=aws_pgvector")
    repository = AwsPgvectorUserProfileRepository(
        AwsPgvectorClient(settings, role="writer")
    )
    updater = UpdateUserInterestProfilesUseCase(
        MainApiFeedInteractionSource(settings),
        repository,
        half_life_days=settings.user_profile_half_life_days,
    )
    last_event_id = await RebuildUserInterestProfilesUseCase(
        updater, repository
    ).execute(user_id=args.user_id, page_limit=args.page_limit)
    print(f"rebuild_user={args.user_id or 'all'} last_event_id={last_event_id}")


if __name__ == "__main__":
    asyncio.run(main())
