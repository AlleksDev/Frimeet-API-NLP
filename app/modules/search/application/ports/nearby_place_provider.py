from typing import Protocol


class NearbyPlaceProvider(Protocol):
    async def get_nearby_place_ids(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int,
    ) -> set[str]:
        """Resolve geographic truth in the main API and return nearby place IDs."""
