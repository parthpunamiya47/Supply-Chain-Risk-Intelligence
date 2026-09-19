"""SupplyChain Sentinel - Stage 4: recommendations API router.

POST /api/recommendations - two mutually exclusive modes:

1. {"features": {...}}            -> Stage 1 ML prediction drives the context
2. {"node_id": ..., "risk_probability": ...} -> Stage 3 propagation drives it:
   supplier/transport risk are taken from the DOMINANT propagation path ending
   at the affected node, inventory/demand/lead time from the downstream node's
   profile. A "graph event" at a supplier can therefore escalate warehouse /
   customer-level recommendations.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.graph import NODES
from backend.graph_schemas import PathEdge  # noqa: F401  (re-export clarity)
from backend.propagation import propagate
from backend.recommendations import generate_recommendations
from backend.recommender_schemas import (
    RecommendationByGraphEventRequest,
    RecommendationByShipmentRequest,
    RecommendationResponse,
)

router = APIRouter(prefix="/api", tags=["recommendations"])

# Rough per-type inventory/demand/lead-time profiles for graph nodes without
# direct logistics numbers (documented as defaults, not measurements).
NODE_PROFILES: dict[str, dict[str, float]] = {
    "supplier": {"inventory_ratio": 1.2, "demand": 6000, "lead_time": 22.0},
    "factory": {"inventory_ratio": 0.9, "demand": 8000, "lead_time": 18.0},
    "port": {"inventory_ratio": 1.5, "demand": 9000, "lead_time": 25.0},
    "route": {"inventory_ratio": 1.5, "demand": 7000, "lead_time": 24.0},
    "warehouse": {"inventory_ratio": 1.1, "demand": 4500, "lead_time": 8.0},
    "customer": {"inventory_ratio": 0.8, "demand": 3000, "lead_time": 5.0},
}

# Map 0-100 risk band -> rule-severity context (documented banding).
def _band(risk01: float) -> str:
    pct = risk01 * 100
    if pct >= 70:
        return "HIGH"
    if pct >= 40:
        return "MEDIUM"
    return "LOW"


@router.post("/recommendations", response_model=RecommendationResponse)
def recommendations(body: RecommendationByShipmentRequest | RecommendationByGraphEventRequest):
    """Generate mitigation recommendations from a shipment OR a graph event."""
    if body.__class__.__name__ == "RecommendationByShipmentRequest":
        return RecommendationResponse(**_from_shipment(body))
    return RecommendationResponse(**_from_graph_event(body))


# --------------------------------------------------------------------------- #
# Mode 1: shipment features -> ML prediction
# --------------------------------------------------------------------------- #


def _from_shipment(body: RecommendationByShipmentRequest) -> dict[str, Any]:
    from ml.predict import explain_factors, predict_risk

    payload = body.features.model_dump(exclude_none=True)
    try:
        prediction = predict_risk(payload, threshold=body.threshold)
        factors = explain_factors(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    supplier_risk = _supplier_risk_from_features(payload)
    transport_risk = float(payload["transportation_risk"]) * 10.0
    inventory_ratio = payload["inventory_level"] / max(payload["demand"], 1)

    ctx = _base_context(
        source="shipment",
        node_id=payload.get("supplier_id"),
        overall=prediction["risk_probability"],
        level=prediction["risk_level"],
        supplier_risk=supplier_risk,
        transport_risk=transport_risk,
        inventory_ratio=inventory_ratio,
        demand=float(payload["demand"]),
        lead_time=float(payload["average_lead_time"]),
        factors=factors,
    )
    ctx["has_alternates"] = True  # shipment context: network-wide alternates allowed
    ctx["alternates"] = _alternate_suppliers(None)
    return generate_recommendations(ctx)


# --------------------------------------------------------------------------- #
# Mode 2: graph event -> propagation
# --------------------------------------------------------------------------- #


def graph_event_recommendations(
    node_id: str,
    risk_probability: float,
    graph: Any | None = None,
    assess_node: str | None = None,
    overrides: dict[str, Any] | None = None,
    source_label: str | None = None,
) -> dict[str, Any]:
    """Run propagation from an event node and generate recommendations.

    Shared engine for Stage 4's graph mode and the Stage 5 simulator:
      - propagates `risk_probability` from `node_id` through `graph`
        (defaults to the real demo network; the simulator passes a COPY),
      - assesses the most-affected downstream node, or `assess_node` when it
        appears in the affected set,
      - context values (supplier/transport risk, inventory, demand, lead time)
        can be overridden via `overrides` (scenario-adjusted economics),
      - `source_label` customises the explanation phrase (e.g. scenario name).

    Raises KeyError for unknown nodes, ValueError when nothing is reachable.
    """
    from backend.graph import build_graph

    g = graph if graph is not None else build_graph()
    if node_id not in g:
        raise KeyError(f"Unknown node_id: {node_id!r}")

    propagation = propagate(node_id, risk_probability, g=g)
    if not propagation["affected_nodes"]:
        raise ValueError("Event does not reach any downstream node")

    # assess the most-affected downstream node (dominant path end)
    worst = propagation["affected_nodes"][0]
    if assess_node is not None:
        match = next(
            (a for a in propagation["affected_nodes"] if a["node_id"] == assess_node), None
        )
        if match is not None:
            worst = match
    assessed_id = worst["node_id"]
    node_type = worst["type"]

    # risk reaching this node (0-1 -> 0-100)
    propagated = worst["propagated_risk"] * 100.0
    ov = overrides or {}
    supplier_risk = float(ov.get("supplier_risk", propagated))
    transport_risk = float(ov.get("transport_risk", propagated * 0.9))
    profile = NODE_PROFILES.get(node_type, NODE_PROFILES["warehouse"])
    inventory_ratio = float(ov.get("inventory_ratio", profile["inventory_ratio"]))
    demand = float(ov.get("demand", profile["demand"]))
    lead_time = float(ov.get("lead_time", profile["lead_time"]))

    label = source_label or f"Upstream {propagation['source']} risk event"
    factors = [
        {
            "feature": "propagation",
            "phrase": label,
            "value": propagation["original_risk"],
            "healthy_median": 0.0,
            "deviation_iqr": min(worst["hops"], 3),
            "contribution": round(propagated / 100.0, 4),
        }
    ]

    ctx = _base_context(
        source="graph",
        node_id=assessed_id,
        overall=propagated,
        level=_band(propagated / 100.0),
        supplier_risk=supplier_risk,
        transport_risk=transport_risk,
        inventory_ratio=inventory_ratio,
        demand=demand,
        lead_time=lead_time,
        factors=factors,
    )
    # alternates must feed the SAME factory as the event source (if source is a
    # supplier); otherwise network-wide alternates apply. Uses the static
    # network relationships - documented, not scenario-dependent.
    source_type = g.nodes[node_id].get("type")
    ctx["has_alternates"] = True
    ctx["alternates"] = (
        _alternate_suppliers_for_source(node_id)
        if source_type == "supplier"
        else _alternate_suppliers(None)
    )
    ctx["propagation_path"] = worst["propagation_path"]
    ctx["hops"] = worst["hops"]
    return generate_recommendations(ctx)


def _from_graph_event(body: RecommendationByGraphEventRequest) -> dict[str, Any]:
    try:
        return graph_event_recommendations(body.node_id, body.risk_probability)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _alternate_suppliers_for_source(source_id: str) -> list[dict[str, Any]]:
    """Alternate suppliers feeding the same factory as `source_id`."""
    return _alternate_suppliers(source_id)


def _supplier_risk_from_features(payload: dict[str, Any]) -> float:
    """Combine reliability, past delays and historical disruptions into a
    transparent 0-100 supplier-risk score (documented heuristic, mirrors the
    generator's dominant coefficients).

    Weights: reliability gap 0.45, past delays 0.35, historical disruptions 0.20.
    A supplier with reliability 50/100, 13/14 delays and 6/7 disruptions
    scores ~72 -> crosses the 70 action threshold, matching the ML model's
    HIGH verdict for such shipments.
    """
    reliability_gap = max(0.0, (100.0 - float(payload["supplier_reliability"])) / 100.0)  # 0-1
    delays = min(float(payload["previous_delays"]) / 14.0, 1.0)                            # 0-1
    hist = min(float(payload["historical_disruptions"]) / 7.0, 1.0)                        # 0-1
    score = 100.0 * (0.45 * reliability_gap + 0.35 * delays + 0.20 * hist)
    return min(score, 100.0)


def _base_context(**kwargs: Any) -> dict[str, Any]:
    ctx = {
        "source": kwargs["source"],
        "node_id": kwargs.get("node_id"),
        "overall_risk": kwargs["overall"],
        "risk_level": kwargs["level"],
        "supplier_risk": kwargs["supplier_risk"],
        "transport_risk": kwargs["transport_risk"],
        "inventory_ratio": kwargs["inventory_ratio"],
        "demand": kwargs["demand"],
        "lead_time": kwargs["lead_time"],
        "factors": kwargs["factors"],
        "high_fired": False,
    }
    return ctx


def _alternate_suppliers(node_id: str | None) -> list[dict[str, Any]]:
    from backend.recommendations import _alternate_suppliers as _impl

    return _impl(node_id)
