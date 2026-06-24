from fastapi import APIRouter, Depends

from app.modules.posts.api.dependencies import (
    get_post_clusters_use_case,
    get_recommend_posts_use_case,
)
from app.modules.posts.api.schemas import (
    PostClustersResponse,
    PostRecommendationRequest,
    PostRecommendationResponse,
    cluster_to_schema,
    post_to_schema,
)
from app.modules.posts.application.use_cases.get_post_clusters import GetPostClustersUseCase
from app.modules.posts.application.use_cases.recommend_posts import RecommendPostsUseCase
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
    result = await use_case.execute(
        query=payload.query,
        city=payload.city,
        limit=payload.limit,
    )
    return PostRecommendationResponse(
        query=result.query,
        posts=[post_to_schema(post) for post in result.posts],
        metadata=result.metadata,
    )


@router.get("/clusters", response_model=PostClustersResponse)
async def get_post_clusters(
    use_case: GetPostClustersUseCase = Depends(get_post_clusters_use_case),
) -> PostClustersResponse:
    result = await use_case.execute()
    return PostClustersResponse(
        clusters=[cluster_to_schema(cluster) for cluster in result.clusters],
        metadata=result.metadata,
    )
