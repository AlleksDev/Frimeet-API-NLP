from datetime import UTC, datetime
from math import sqrt

from app.modules.posts.domain.ports.feed_interaction_source import FeedInteractionSource
from app.modules.posts.domain.ports.user_profile_repository import UserProfileRepository
from app.modules.posts.domain.profiles import FeedInteraction, UserInterestState


class UpdateUserInterestProfilesUseCase:
    CONSUMER = "user-interest-profiles-v1"
    PROFILE_VERSION = "user-interest-v1"
    BASE_WEIGHTS = {
        "save": 1.00,
        "share": 0.90,
        "comment": 0.75,
        "like": 0.60,
        "open": 0.35,
        "impression": 0.05,
        "hide": -0.80,
        "not_interested": -1.00,
        "unsave": -1.00,
        "unlike": -0.60,
    }

    def __init__(
        self,
        source: FeedInteractionSource,
        repository: UserProfileRepository,
        half_life_days: float = 30.0,
    ) -> None:
        self._source = source
        self._repository = repository
        self._half_life_days = half_life_days

    async def execute(
        self, page_limit: int = 1000, user_id: str | None = None
    ) -> int:
        consumer = self.consumer_for(user_id)
        after_id = await self._repository.get_checkpoint(consumer)
        batch: list[FeedInteraction] = []
        last_event_id = after_id
        async for interaction in self._source.iter_interactions(
            after_id, page_limit, user_id
        ):
            batch.append(interaction)
            last_event_id = interaction.event_id
            if len(batch) >= page_limit:
                await self._process_batch(batch)
                await self._repository.save_checkpoint(consumer, last_event_id)
                batch = []
        if batch:
            await self._process_batch(batch)
            await self._repository.save_checkpoint(consumer, last_event_id)
        return last_event_id

    async def _process_batch(self, interactions: list[FeedInteraction]) -> None:
        unsupported = sorted(
            {item.event_type for item in interactions} - set(self.BASE_WEIGHTS)
        )
        if unsupported:
            raise ValueError(
                "tipos de interaccion no soportados: " + ",".join(unsupported)
            )
        vectors = await self._repository.get_post_embeddings(
            {item.post_id for item in interactions}
        )
        states = await self._repository.get_states(
            {item.user_id for item in interactions}
        )
        missing = sorted({item.post_id for item in interactions} - set(vectors))
        if missing:
            raise RuntimeError(
                "faltan embeddings para interacciones: " + ",".join(missing[:20])
            )
        now = datetime.now(UTC)
        for item in interactions:
            vector = vectors.get(item.post_id)
            if vector is None or self.BASE_WEIGHTS.get(item.event_type, 0.0) == 0.0:
                continue
            weight = self._weight(item, now)
            state = states.get(item.user_id)
            if state is not None and item.event_id <= state.last_event_id:
                continue
            if state is None:
                if weight <= 0:
                    continue
                state = UserInterestState(
                    user_id=item.user_id,
                    weighted_sum=[0.0] * len(vector),
                    total_weight=0.0,
                    interaction_count=0,
                    last_event_id=0,
                    embedding=[0.0] * len(vector),
                )
                states[item.user_id] = state
            if len(state.weighted_sum) != len(vector):
                raise ValueError("dimension incompatible al actualizar perfil de usuario")
            state.weighted_sum = [
                current + weight * value
                for current, value in zip(state.weighted_sum, vector)
            ]
            state.total_weight += abs(weight)
            state.interaction_count += 1
            state.last_event_id = max(state.last_event_id, item.event_id)
            state.embedding = _normalize(state.weighted_sum)
        await self._repository.save_states(list(states.values()), self.PROFILE_VERSION)

    def consumer_for(self, user_id: str | None) -> str:
        return self.CONSUMER if user_id is None else f"{self.CONSUMER}:user:{user_id}"

    def _weight(self, interaction: FeedInteraction, now: datetime) -> float:
        base = self.BASE_WEIGHTS.get(interaction.event_type, 0.0)
        if interaction.event_type == "open" and (interaction.dwell_time_ms or 0) < 3_000:
            base = 0.10
        age_days = max(0.0, (now - interaction.occurred_at).total_seconds() / 86_400)
        return base * (0.5 ** (age_days / self._half_life_days))


def _normalize(vector: list[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else [0.0] * len(vector)
