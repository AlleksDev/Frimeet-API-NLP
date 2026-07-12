from app.modules.posts.application.use_cases.update_user_interest_profiles import (
    UpdateUserInterestProfilesUseCase,
)
from app.modules.posts.domain.ports.user_profile_repository import UserProfileRepository


class RebuildUserInterestProfilesUseCase:
    def __init__(
        self,
        updater: UpdateUserInterestProfilesUseCase,
        repository: UserProfileRepository,
    ) -> None:
        self._updater = updater
        self._repository = repository

    async def execute(
        self, user_id: str | None = None, page_limit: int = 1000
    ) -> int:
        consumer = self._updater.consumer_for(user_id)
        await self._repository.reset_profiles(user_id)
        await self._repository.reset_checkpoint(consumer)
        return await self._updater.execute(page_limit=page_limit, user_id=user_id)
