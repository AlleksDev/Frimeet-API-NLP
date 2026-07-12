from datetime import UTC, datetime

import pytest

from app.modules.posts.application.use_cases.update_user_interest_profiles import UpdateUserInterestProfilesUseCase
from app.modules.posts.domain.profiles import FeedInteraction


class InteractionSourceStub:
    async def iter_interactions(self, after_id, page_limit=1000, user_id=None):
        yield FeedInteraction(1, "u1", "p1", "save", datetime.now(UTC))
        yield FeedInteraction(2, "u1", "p2", "hide", datetime.now(UTC))


class ProfileRepositoryStub:
    def __init__(self):
        self.states = {}
        self.checkpoint = 0

    async def get_post_embeddings(self, post_ids):
        return {"p1": [1.0, 0.0], "p2": [0.0, 1.0]}

    async def get_states(self, user_ids):
        return self.states

    async def save_states(self, states, profile_version):
        self.states = {state.user_id: state for state in states}

    async def get_checkpoint(self, consumer):
        return self.checkpoint

    async def save_checkpoint(self, consumer, event_id):
        self.checkpoint = event_id

    async def reset_profiles(self, user_id=None):
        self.states = {}

    async def reset_checkpoint(self, consumer):
        self.checkpoint = 0


class UnsupportedInteractionSourceStub:
    async def iter_interactions(self, after_id, page_limit=1000, user_id=None):
        yield FeedInteraction(1, "u1", "p1", "unknown", datetime.now(UTC))


@pytest.mark.asyncio
async def test_profile_updates_from_positive_and_negative_signals() -> None:
    repository = ProfileRepositoryStub()
    use_case = UpdateUserInterestProfilesUseCase(
        InteractionSourceStub(), repository, half_life_days=30
    )
    assert await use_case.execute() == 2
    state = repository.states["u1"]
    assert state.interaction_count == 2
    assert state.embedding[0] > 0
    assert state.embedding[1] < 0
    assert repository.checkpoint == 2


@pytest.mark.asyncio
async def test_profile_does_not_advance_checkpoint_for_unknown_signal() -> None:
    repository = ProfileRepositoryStub()
    use_case = UpdateUserInterestProfilesUseCase(
        UnsupportedInteractionSourceStub(), repository, half_life_days=30
    )
    with pytest.raises(ValueError, match="tipos de interaccion no soportados"):
        await use_case.execute()
    assert repository.checkpoint == 0
