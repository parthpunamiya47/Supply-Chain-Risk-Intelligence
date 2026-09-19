"""SupplyChain Sentinel - Stage 3: Pydantic schemas for the graph API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PropagationRequest(BaseModel):
    """Body for POST /api/graph/propagate-risk."""

    node_id: str = Field(..., min_length=1, max_length=64, examples=["sup_a"])
    risk_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Original disruption risk at the source node, 0-1 (e.g. 0.85)",
    )


class PathEdge(BaseModel):
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    relationship: str
    weight: float

    model_config = {"populate_by_name": True}


class AlternativePath(BaseModel):
    path: list[str]
    hops: int
    factor: float
    propagated_risk: float


class AffectedNode(BaseModel):
    node_id: str
    name: str
    type: str
    original_risk: float
    propagated_risk: float
    contribution: float
    hops: int
    propagation_path: list[str]
    path_edges: list[PathEdge]
    alternative_paths: list[AlternativePath]


class PropagationSummary(BaseModel):
    nodes_affected: int
    max_propagated_risk: float
    max_hops_to_any_node: int
    customer_nodes_affected: int


class PropagationResponse(BaseModel):
    source: str
    original_risk: float
    parameters: dict[str, float | int]
    affected_nodes: list[AffectedNode]
    summary: PropagationSummary


class GraphNodeData(BaseModel):
    label: str
    type: str
    risk: float
    capacity: int | None = None
    location: str | None = None


class GraphNode(BaseModel):
    id: str
    position: dict[str, float]
    data: GraphNodeData
    type: str = "custom"


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str
    data: dict[str, float | str]
    animated: bool = False


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    tiers: dict[str, list[str]]
    width: float
    height: float
    node_count: int
    edge_count: int
