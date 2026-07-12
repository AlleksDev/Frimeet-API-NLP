from fastapi import APIRouter, Depends

from app.modules.posts.application.use_cases.recommend_posts import RecommendPostsUseCase
from app.modules.posts.infrastructure.api.public_dependencies import get_recommend_posts_use_case
from app.modules.posts.infrastructure.api.public_schemas import (
    PostRecommendationRequest,
    PostRecommendationResponse,
    post_to_schema,
)
from app.shared.security.rate_limit import rate_limit_placeholder

router = APIRouter(
    prefix="/posts",
    tags=["posts"],
    dependencies=[Depends(rate_limit_placeholder)],
)


@router.post("/recommendations", response_model=PostRecommendationResponse)
async def recommend_posts(
    payload: PostRecommendationRequest,
    use_case: RecommendPostsUseCase = Depends(get_recommend_posts_use_case),
) -> PostRecommendationResponse:
    result = await use_case.execute(payload.query, payload.city, payload.limit)
    return PostRecommendationResponse(
        query=result.query,
        posts=[post_to_schema(post) for post in result.posts],
        metadata=result.metadata,
    )
