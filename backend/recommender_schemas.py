"""SupplyChain Sentinel - Stage 4: Pydantic schemas for the recommendations API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from backend.schemas import ShipmentFeatures  # reuse the validated feature model


class RecommendationItem(BaseModel):
    rule_id: str
    action: str
    reason: str
    risk_reduction: float = Field(..., ge=0, le=100)
    priority: str = Field(..., pattern="^(HIGH|MEDIUM|LOW|NONE)$")
    cost_estimate: str = Field(..., pattern="^(NONE|LOW|MEDIUM|HIGH)$")
    cost_note: str
    details: dict[str, object] = {}


class RecommendationContext(BaseModel):
    source: str = Field(..., pattern="^(shipment|graph)$")
    node_id: str | None = None
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH)$")
    overall_risk: float
    supplier_risk: float
    transport_risk: float
    inventory_ratio: float
    demand: float
    lead_time_days: float
    top_risk_factors: list[str]


class RecommendationSummary(BaseModel):
    rules_fired: list[str]
    count: int
    highest_priority: str
    total_potential_risk_reduction: float


class RecommendationResponse(BaseModel):
    context: RecommendationContext
    recommendations: list[RecommendationItem]
    summary: RecommendationSummary


class RecommendationByShipmentRequest(BaseModel):
    """Recommendations for one shipment (runs the Stage 1 model first).

    extra="forbid" rejects mixed-mode payloads (features + node_id together).
    """

    model_config = ConfigDict(extra="forbid")

    features: ShipmentFeatures
    threshold: float = Field(default=0.5, gt=0, lt=1)


class RecommendationByGraphEventRequest(BaseModel):
    """Recommendations for a graph risk event (runs Stage 3 propagation)."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(..., min_length=1, max_length=64)
    risk_probability: float = Field(..., ge=0, le=1)
