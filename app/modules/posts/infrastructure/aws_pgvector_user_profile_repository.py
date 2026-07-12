import json
from collections.abc import Iterable

from app.modules.posts.domain.ports.user_profile_repository import UserProfileRepository
from app.modules.posts.domain.profiles import UserInterestState
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient
from app.shared.vector_store.sql import vector_literal


class AwsPgvectorUserProfileRepository(UserProfileRepository):
    def __init__(self, client: AwsPgvectorClient) -> None:
        self._client = client

    async def get_post_embeddings(self, post_ids: Iterable[str]) -> dict[str, list[float]]:
        ids = list(post_ids)
        if not ids:
            return {}
        async with self._client.connection() as connection:
            rows = await connection.fetch(
                "SELECT * FROM get_post_embeddings_for_profile($1::text[])", ids
            )
        return {str(row["post_id"]): _parse_vector(row["embedding"]) for row in rows}

    async def get_states(self, user_ids: Iterable[str]) -> dict[str, UserInterestState]:
        ids = list(user_ids)
        if not ids:
            return {}
        async with self._client.connection() as connection:
            rows = await connection.fetch(
                "SELECT * FROM get_user_interest_states($1::text[])", ids
            )
        return {
            str(row["user_id"]): UserInterestState(
                user_id=str(row["user_id"]),
                weighted_sum=_parse_vector(row["weighted_sum"]),
                total_weight=float(row["total_weight"]),
                interaction_count=int(row["interaction_count"]),
                last_event_id=int(row["last_event_id"]),
                embedding=_parse_vector(row["embedding"]),
            )
            for row in rows
        }

    async def save_states(self, states: list[UserInterestState], profile_version: str) -> None:
        if not states:
            return
        query = """
            SELECT upsert_user_interest_state(
                $1::text, $2::vector, $3::vector, $4::double precision,
                $5::integer, $6::bigint, $7::text, $8::text
            )
        """
        values = [
            (
                state.user_id,
                vector_literal(state.embedding),
                vector_literal(state.weighted_sum),
                state.total_weight,
                state.interaction_count,
                state.last_event_id,
                profile_version,
                _state_hash(state),
            )
            for state in states
        ]
        async with self._client.connection() as connection:
            await connection.executemany(query, values)

    async def get_checkpoint(self, consumer: str) -> int:
        async with self._client.connection() as connection:
            value = await connection.fetchval(
                "SELECT get_post_sync_checkpoint($1::text)", consumer
            )
        return int(value or 0)

    async def save_checkpoint(self, consumer: str, event_id: int) -> None:
        async with self._client.connection() as connection:
            await connection.execute(
                "SELECT save_post_sync_checkpoint($1::text, $2::bigint)",
                consumer,
                event_id,
            )

    async def reset_profiles(self, user_id: str | None = None) -> None:
        async with self._client.connection() as connection:
            await connection.execute(
                "SELECT reset_user_interest_profiles($1::text)", user_id
            )

    async def reset_checkpoint(self, consumer: str) -> None:
        async with self._client.connection() as connection:
            await connection.execute(
                "SELECT reset_post_sync_checkpoint($1::text)", consumer
            )


def _parse_vector(value: object) -> list[float]:
    return [float(item) for item in str(value).strip("[]").split(",") if item]


def _state_hash(state: UserInterestState) -> str:
    from hashlib import sha256

    payload = json.dumps(
        [state.last_event_id, state.interaction_count, state.total_weight],
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()
