from fastapi import Depends, Request

from app.controllers import base
from app.controllers.v1.base import new_router
from app.models.schema import (
    NicheQueriesRequest,
    NicheQueriesResponse,
    NicheScanRequest,
    NicheScanResponse,
)
from app.services import niche_finder
from app.utils import utils

# Те же правила аутентификации, что и у остальных v1-эндпоинтов:
# при пустом api_key verify_token пропускает запросы без токена.
router = new_router(dependencies=[Depends(base.verify_token)])


@router.post(
    "/niche/scan",
    response_model=NicheScanResponse,
    summary="Scan YouTube for rising videos from small channels",
)
def scan_niche(request: Request, body: NicheScanRequest):
    result = niche_finder.find_rising_videos(
        queries=body.queries,
        days=body.days,
        max_subscribers=body.max_subscribers,
        min_views=body.min_views,
        per_query=body.per_query,
    )
    return utils.get_response(200, result)


@router.post(
    "/niche/queries",
    response_model=NicheQueriesResponse,
    summary="Suggest YouTube search queries for a topic",
)
def suggest_niche_queries(request: Request, body: NicheQueriesRequest):
    queries = niche_finder.suggest_queries(
        topic=body.topic, language=body.language, count=body.count
    )
    return utils.get_response(200, {"queries": queries})
