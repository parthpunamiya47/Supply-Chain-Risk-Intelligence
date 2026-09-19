"""SupplyChain Sentinel - Stage 3: graph API router.

  GET  /api/graph                    -> full network as React Flow JSON
  POST /api/graph/propagate-risk     -> explainable downstream risk propagation
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.graph import build_graph, to_react_flow
from backend.graph_schemas import GraphResponse, PropagationRequest, PropagationResponse
from backend.propagation import propagate

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("", response_model=GraphResponse)
def get_graph() -> GraphResponse:
    """Return the complete demo network, laid out in tiers, React Flow ready."""
    return GraphResponse(**to_react_flow())


@router.post("/propagate-risk", response_model=PropagationResponse)
def propagate_risk(body: PropagationRequest) -> PropagationResponse:
    """Propagate a risk event from `node_id` through the dependency network.

    propagated(n) = original_risk * prod(weight_e * DECAY) along each path;
    each node reports the dominant (max) path plus all alternatives.
    """
    from backend.graph import NODES

    if not any(n["id"] == body.node_id for n in NODES):
        raise HTTPException(status_code=404, detail=f"Unknown node_id: {body.node_id!r}")
    try:
        result = propagate(body.node_id, body.risk_probability)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PropagationResponse(**result)
