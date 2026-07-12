from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.shared.security import rate_limit


def _request(path: str = "/internal/search/candidates") -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
    )


@pytest.mark.asyncio
async def test_internal_rate_limit_rejects_requests_over_the_window(monkeypatch) -> None:
    settings = SimpleNamespace(
        rate_limit_window_seconds=60,
        rate_limit_requests_per_window=10,
        internal_rate_limit_requests_per_window=1,
    )
    monkeypatch.setattr(rate_limit, "get_settings", lambda: settings)
    rate_limit.reset_rate_limit_state()

    await rate_limit.rate_limit_placeholder(_request())
    with pytest.raises(HTTPException) as captured:
        await rate_limit.rate_limit_placeholder(_request())

    assert captured.value.status_code == 429
    assert captured.value.headers["Retry-After"]
    rate_limit.reset_rate_limit_state()
