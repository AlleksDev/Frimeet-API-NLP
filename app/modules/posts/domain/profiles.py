from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FeedInteraction:
    event_id: int
    user_id: str
    post_id: str
    event_type: str
    occurred_at: datetime
    dwell_time_ms: int | None = None


@dataclass
class UserInterestState:
    user_id: str
    weighted_sum: list[float]
    total_weight: float
    interaction_count: int
    last_event_id: int
    embedding: list[float]
