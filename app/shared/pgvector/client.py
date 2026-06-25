from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import ssl

import asyncpg

from app.shared.config.settings import Settings


class PgVectorConnectionFactory:
    def __init__(self, settings: Settings) -> None:
        if not settings.pgvector_dsn:
            raise RuntimeError("PGVECTOR_DSN is required when VECTOR_STORE_MODE=pgvector")
        self._dsn = settings.pgvector_dsn
        self._ssl = _build_ssl_context(settings.pgvector_ssl)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        connection = await asyncpg.connect(dsn=self._dsn, ssl=self._ssl)
        try:
            yield connection
        finally:
            await connection.close()


def _build_ssl_context(mode: str | None) -> ssl.SSLContext | bool | None:
    if mode is None or mode in {"", "disable", "false", "0"}:
        return None
    if mode in {"prefer", "allow"}:
        return None
    if mode in {"require", "true", "1"}:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    return True
