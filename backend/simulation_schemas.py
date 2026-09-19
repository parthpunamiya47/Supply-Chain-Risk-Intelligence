"""SupplyChain Sentinel - Stage 5: Pydantic schemas for the simulation API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SimulationRequest(BaseModel):
    """Body for POST /api/simulation.

    Only `scenario` is required. `target_node` defaults to the scenario's
    demo node; `event_risk` and `demand_multiplier` override the presets.
    """

    model_config = ConfigDict(extra="forbid")

    scenario: str = Field(..., min_length=1, max_length=64, examples=["supplier_failure"])
    target_node: str | None = Field(default=None, min_length=1, max_length=64)
    event_risk: float | None = Field(default=None, gt=0, le=1,
                                     description="Override the scenario's event risk (0-1)")
    demand_multiplier: float | None = Field(
        default=None, gt=0, description="Scale the scenario's demand (e.g. 1.5 = +50%)"
    )


class AffectedNodeDelta(BaseModel):
    node_id: str
    name: str
    type: str
    before_risk: float
    after_risk: float
    delta: float
    hops: int


class PathEdgeOut(BaseModel):
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    relationship: str
    weight: float

    model_config = {"populate_by_name": True}


class AlternativePathOut(BaseModel):
    path: list[str]
    hops: int
    factor: float
    propagated_risk: float


class PropagationPathOut(BaseModel):
    node_id: str
    dominant_path: list[str]
    path_edges: list[PathEdgeOut]
    alternative_paths: list[AlternativePathOut]


class RecommendationItemOut(BaseModel):
    rule_id: str
    action: str
    reason: str
    risk_reduction: float
    priority: str
    cost_estimate: str
    cost_note: str
    details: dict[str, object] = {}


class RecommendationContextOut(BaseModel):
    source: str
    node_id: str | None
    risk_level: str
    overall_risk: float
    supplier_risk: float
    transport_risk: float
    inventory_ratio: float
    demand: float
    lead_time_days: float
    top_risk_factors: list[str]


class SimulationSummary(BaseModel):
    nodes_affected: int
    max_delta: float
    customers_affected: int
    rules_fired: list[str]
    highest_priority: str
    graph_modified: bool
    note: str


class SimulationResponse(BaseModel):
    scenario: str
    scenario_label: str
    description: str
    target_node: str
    event_risk: float
    before_risk: dict[str, float]
    after_risk: dict[str, float]
    affected_nodes: list[AffectedNodeDelta]
    propagation_paths: list[PropagationPathOut]
    recommendations: list[RecommendationItemOut]
    recommendation_context: RecommendationContextOut
    summary: SimulationSummary


class ScenarioInfo(BaseModel):
    scenario: str
    label: str
    description: str
    default_target: str
    target_type: str
    event_risk: float


class ScenarioListResponse(BaseModel):
    scenarios: list[ScenarioInfo]
