from fastapi import APIRouter

from app.modules.posts.infrastructure.api.internal_router import router as internal_router
from app.modules.posts.infrastructure.api.public_router import router as public_router

router = APIRouter()
router.include_router(public_router)
router.include_router(internal_router)
