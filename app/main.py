from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.modules.places.api.router import router as places_router
from app.modules.posts.api.router import router as posts_router
from app.shared.config.settings import get_settings
from app.shared.errors.exceptions import AppError
from app.shared.errors.handlers import app_error_handler
from app.shared.logging.config import configure_logging
from app.shared.security.request_limits import RequestSizeLimitMiddleware
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Independent NLP Service",
        version="0.1.0",
        description="NLP backend for semantic search, recommendations, embeddings, AWS pgvector and Llama/Groq response drafting.",
    )
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_body_size=settings.max_request_body_bytes,
    )
    app.add_exception_handler(AppError, app_error_handler)

    @app.get("/", tags=["system"])
    async def root() -> dict[str, object]:
        return {
            "service": "frimeet-api-nlp",
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
            "ready": "/ready",
        }

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["system"], response_model=None)
    async def ready() -> dict[str, object] | JSONResponse:
        vector_store = {
            "provider": settings.vector_store_provider,
            "host": settings.pgvector_host,
            "port": settings.pgvector_port,
            "database": settings.pgvector_database,
            "ssl_mode": settings.pgvector_ssl_mode,
        }
        is_ready = True

        if settings.vector_store_provider == "aws_pgvector":
            try:
                contract = await AwsPgvectorClient(settings, role="reader").check_read_contract()
            except Exception as exc:
                contract = {
                    "ready": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            vector_store["contract"] = contract
            is_ready = bool(contract.get("ready"))

        payload = {
            "status": "ready" if is_ready else "not_ready",
            "environment": settings.env,
            "dependencies": {
                "vector_store": vector_store,
                "llm": {
                    "provider": "groq" if settings.groq_api_key else "mock",
                    "model": settings.groq_model,
                },
            },
        }
        if not is_ready:
            return JSONResponse(status_code=503, content=payload)
        return payload

    app.include_router(places_router)
    app.include_router(posts_router)
    return app


app = create_app()
