"""SupplyChain Sentinel - Stage 6: overview API router (GET /api/overview)."""

from __future__ import annotations

from fastapi import APIRouter

from backend.overview import build_overview
from backend.overview_schemas import OverviewResponse

router = APIRouter(prefix="/api", tags=["overview"])


@router.get("/overview", response_model=OverviewResponse)
def get_overview() -> OverviewResponse:
    """Real dashboard KPIs derived from the dataset, model and graph."""
    return OverviewResponse(**build_overview())
