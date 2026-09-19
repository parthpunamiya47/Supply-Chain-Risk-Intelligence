"""SupplyChain Sentinel - Stage 6: Pydantic schemas for the overview API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AlertItem(BaseModel):
    id: str
    node_id: str
    name: str
    type: str
    risk: float
    risk_probability: float
    severity: str
    message: str


class SupplierPerformanceItem(BaseModel):
    supplier_id: str
    shipments: int
    disruptions: int
    disruption_rate_pct: float
    avg_reliability: float


class TrendPoint(BaseModel):
    period: str
    disruption_rate_pct: float
    shipments: int


class OverviewResponse(BaseModel):
    total_suppliers: int
    graph_nodes: int
    high_risk_nodes: int
    risk_distribution: dict[str, int]
    prediction_distribution: dict[str, int] = {}
    alerts: list[AlertItem]
    at_risk_shipments: int = 0
    average_risk: float = 0.0
    sample_size: int = 0
    supplier_performance: list[SupplierPerformanceItem] = []
    risk_trend: list[TrendPoint] = []
    note: str | None = None
