from typing import Protocol


class NearbyPlaceProvider(Protocol):
    async def get_nearby_place_ids(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int,
    ) -> set[str]:
        """Return place IDs accepted by the main API geographic filter."""
