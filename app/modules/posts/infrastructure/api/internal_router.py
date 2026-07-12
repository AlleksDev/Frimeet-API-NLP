import hmac
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, status

from app.modules.posts.application.use_cases.rank_user_feed import RankUserFeedUseCase
from app.modules.posts.application.use_cases.activate_post_cluster_run import ActivatePostClusterRunUseCase
from app.modules.posts.application.use_cases.get_post_cluster_status import GetPostClusterStatusUseCase
from app.modules.posts.application.use_cases.get_post_cluster_run import GetPostClusterRunUseCase
from app.modules.posts.application.use_cases.train_post_clusters import TrainPostClustersUseCase
from app.modules.posts.infrastructure.api.dependencies import (
    get_activate_cluster_use_case,
    get_cluster_status_use_case,
    get_cluster_run_use_case,
    get_rank_user_feed_use_case,
    get_train_post_clusters_use_case,
)
from app.modules.posts.infrastructure.api.schemas import (
    ClusterActivationResponse,
    ClusterRunDetailResponse,
    ClusterTrainingScheduledResponse,
    ClusterStatusResponse,
    RankFeedRequest,
    RankFeedResponse,
    to_domain,
    to_response,
)
from app.shared.config.settings import get_settings

router = APIRouter(prefix="/internal/posts", tags=["internal-posts"])
logger = logging.getLogger(__name__)


def require_service_token(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    expected = settings.post_feed_internal_token
    if not expected:
        if settings.env == "local":
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="POST_FEED_INTERNAL_TOKEN no configurado",
        )
    scheme, separator, received = (authorization or "").partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not received:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    received = received.strip()
    if not hmac.compare_digest(received, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@router.post(
    "/feed/rank",
    response_model=RankFeedResponse,
    dependencies=[Depends(require_service_token)],
)
async def rank_feed(
    payload: RankFeedRequest,
    use_case: RankUserFeedUseCase = Depends(get_rank_user_feed_use_case),
) -> RankFeedResponse:
    post_ids = [candidate.post_id for candidate in payload.candidate_posts]
    if len(post_ids) != len(set(post_ids)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="candidate_posts contiene IDs duplicados",
        )
    ranking = await use_case.execute(
        user_id=payload.user_id,
        candidates=[to_domain(candidate) for candidate in payload.candidate_posts],
        snapshot_at=payload.snapshot_at,
        result_limit=payload.result_limit,
    )
    logger.info(
        "feed_rank candidates=%s results=%s cold_start=%s missing_embeddings=%s "
        "duplicate_penalized=%s diversity_relaxations=%s cluster_run_id=%s",
        len(payload.candidate_posts),
        len(ranking.items),
        ranking.cold_start,
        ranking.missing_embedding_count,
        ranking.duplicate_penalized_count,
        ranking.diversity_relaxations,
        ranking.cluster_run_id,
    )
    return to_response(ranking)


@router.post(
    "/clusters/runs",
    response_model=ClusterTrainingScheduledResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_service_token)],
)
async def train_clusters(
    background_tasks: BackgroundTasks,
    use_case: TrainPostClustersUseCase = Depends(get_train_post_clusters_use_case),
) -> ClusterTrainingScheduledResponse:
    background_tasks.add_task(use_case.execute)
    return ClusterTrainingScheduledResponse()


@router.get(
    "/clusters/status",
    response_model=ClusterStatusResponse,
    dependencies=[Depends(require_service_token)],
)
async def cluster_status(
    use_case: GetPostClusterStatusUseCase = Depends(get_cluster_status_use_case),
) -> ClusterStatusResponse:
    return ClusterStatusResponse(**(await use_case.execute()).__dict__)


@router.get(
    "/clusters/runs/{run_id}",
    response_model=ClusterRunDetailResponse,
    dependencies=[Depends(require_service_token)],
)
async def cluster_run_detail(
    run_id: str,
    use_case: GetPostClusterRunUseCase = Depends(get_cluster_run_use_case),
) -> ClusterRunDetailResponse:
    result = await use_case.execute(run_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return ClusterRunDetailResponse(**result.__dict__)


@router.post(
    "/clusters/runs/{run_id}/activate",
    response_model=ClusterActivationResponse,
    dependencies=[Depends(require_service_token)],
)
async def activate_cluster_run(
    run_id: str,
    use_case: ActivatePostClusterRunUseCase = Depends(get_activate_cluster_use_case),
) -> ClusterActivationResponse:
    await use_case.execute(run_id)
    return ClusterActivationResponse(run_id=run_id)
