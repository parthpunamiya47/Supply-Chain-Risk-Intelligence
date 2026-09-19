"""SupplyChain Sentinel - Stage 5: what-if disruption simulator.

Runs hypothetical scenarios on a COPY of the network - the real graph
(backend/graph.py NODES/EDGES + NetworkX instance) is never mutated:

    1. build_graph()                       -> fresh DiGraph
    2. g.copy()                            -> simulation copy
    3. apply scenario event                -> raises the target node's risk
    4. propagate() on the copy             -> downstream risk spread
    5. merge before/after per node         -> deltas
    6. graph_event_recommendations(...)    -> mitigation actions
    7. return payload with propagation paths

Scenarios (presets in SCENARIOS, all overridable per request):
  supplier_failure        event 0.95 at a supplier node
  port_closure            event 0.95 at a port node
  transportation_disruption event 0.90 at a route node
  demand_spike            milder event 0.55 + demand/inventory overrides
                          (stock pressure) at a warehouse node
"""

from __future__ import annotations

from typing import Any

from backend.graph import NODES, build_graph
from backend.propagation import propagate
from backend.recommender_router import graph_event_recommendations

# --------------------------------------------------------------------------- #
# Scenario presets (all values overridable per request)
# --------------------------------------------------------------------------- #

SCENARIOS: dict[str, dict[str, Any]] = {
    "supplier_failure": {
        "label": "Supplier failure",
        "target_type": "supplier",
        "default_target": "sup_c",
        "event_risk": 0.95,
        "description": "Supplier stops shipping entirely (fire, bankruptcy, strike).",
        "overrides": {"supplier_risk": 95.0, "transport_risk": 60.0},
    },
    "port_closure": {
        "label": "Port closure",
        "target_type": "port",
        "default_target": "port_lb",
        "event_risk": 0.95,
        "description": "Port stops handling vessels (weather, labor action, accident).",
        "overrides": {"transport_risk": 92.0, "supplier_risk": 55.0, "lead_time": 30.0},
    },
    "transportation_disruption": {
        "label": "Transportation disruption",
        "target_type": "route",
        "default_target": "route_rail",
        "event_risk": 0.90,
        "description": "Rail/road lane degraded (derailment, flooding, border closure).",
        "overrides": {"transport_risk": 90.0, "supplier_risk": 45.0, "lead_time": 27.0},
    },
    "demand_spike": {
        "label": "Demand spike",
        "target_type": "warehouse",
        "default_target": "wh_2",
        "event_risk": 0.55,
        "description": "Sudden demand surge drains warehouse stock (promo, season, panic buying).",
        "overrides": {"demand": 18000.0, "inventory_ratio": 0.3, "supplier_risk": 30.0,
                      "transport_risk": 35.0},
    },
}


def _resolve_target(scenario_id: str, target_node: str | None, g) -> str:
    """Pick the request target, the preset default, or any node of the right type."""
    scenario = SCENARIOS[scenario_id]
    if target_node:
        if target_node not in g:
            raise KeyError(f"Unknown target_node: {target_node!r}")
        expected = scenario["target_type"]
        actual = g.nodes[target_node].get("type")
        if actual != expected:
            raise ValueError(
                f"Scenario '{scenario_id}' targets a {expected} node, "
                f"but '{target_node}' is a {actual}."
            )
        return target_node
    if scenario["default_target"] in g:
        return scenario["default_target"]
    matches = [n["id"] for n in NODES if n["type"] == expected]
    if not matches:
        raise ValueError(f"No {expected} node available for scenario '{scenario_id}'")
    return matches[0]


def run_simulation(
    scenario_id: str,
    target_node: str | None = None,
    event_risk: float | None = None,
    demand_multiplier: float | None = None,
) -> dict[str, Any]:
    """Execute one what-if scenario end-to-end on a graph copy."""
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{scenario_id}'. Available: {sorted(SCENARIOS)}"
        )
    scenario = SCENARIOS[scenario_id]

    # 1-2. copy the current network (fresh build + .copy(); the module-level
    # NODES/EDGES and any cached instance are never touched)
    real_graph = build_graph()
    sim_graph = real_graph.copy()

    # 3. resolve + apply the disruption to the COPY
    target = _resolve_target(scenario_id, target_node, sim_graph)
    risk = float(event_risk) if event_risk is not None else float(scenario["event_risk"])
    if not 0.0 < risk <= 1.0:
        raise ValueError("event_risk must be within (0, 1]")

    before_risk: dict[str, float] = {
        n: float(d.get("risk", 0.0)) for n, d in sim_graph.nodes(data=True)
    }
    sim_graph.nodes[target]["risk"] = round(risk, 4)  # only the copy changes

    after_risk: dict[str, float] = dict(before_risk)
    # the event node itself jumps to the event risk (propagation excludes the
    # source, so update it here)
    after_risk[target] = round(max(after_risk[target], risk), 4)

    # 4. propagate through the copy
    propagation = propagate(target, risk, g=sim_graph)

    # 5. affected nodes + before/after merge (after = max(baseline, propagated)
    #    so the simulated stress can only raise a node's effective risk)
    affected: list[dict[str, Any]] = []
    for a in propagation["affected_nodes"]:
        node = a["node_id"]
        after = round(max(after_risk[node], a["propagated_risk"]), 4)
        after_risk[node] = after
        affected.append(
            {
                "node_id": node,
                "name": a["name"],
                "type": a["type"],
                "before_risk": before_risk[node],
                "after_risk": after,
                "delta": round(after - before_risk[node], 4),
                "hops": a["hops"],
            }
        )
    affected.sort(key=lambda x: x["delta"], reverse=True)

    # 6. mitigation recommendations on the same simulated copy
    overrides = dict(scenario.get("overrides", {}))
    if demand_multiplier is not None:
        if demand_multiplier <= 0:
            raise ValueError("demand_multiplier must be positive")
        base_demand = overrides.get("demand", 4500.0)
        overrides["demand"] = round(float(base_demand) * float(demand_multiplier), 1)
    recs = graph_event_recommendations(
        target,
        risk,
        graph=sim_graph,
        overrides=overrides or None,
        source_label=f"{scenario['label']} scenario ({target})",
    )

    # 7. response
    return {
        "scenario": scenario_id,
        "scenario_label": scenario["label"],
        "description": scenario["description"],
        "target_node": target,
        "event_risk": round(risk, 4),
        "before_risk": {k: round(v, 4) for k, v in sorted(before_risk.items())},
        "after_risk": {k: round(v, 4) for k, v in sorted(after_risk.items())},
        "affected_nodes": affected,
        "propagation_paths": [
            {
                "node_id": a["node_id"],
                "dominant_path": a["propagation_path"],
                "path_edges": a["path_edges"],
                "alternative_paths": a["alternative_paths"],
            }
            for a in propagation["affected_nodes"]
        ],
        "recommendations": recs["recommendations"],
        "recommendation_context": recs["context"],
        "summary": {
            "nodes_affected": len(affected),
            "max_delta": affected[0]["delta"] if affected else 0.0,
            "customers_affected": sum(1 for a in affected if a["type"] == "customer"),
            "rules_fired": recs["summary"]["rules_fired"],
            "highest_priority": recs["summary"]["highest_priority"],
            "graph_modified": False,
            "note": "Simulation ran on an in-memory copy; the real network is unchanged.",
        },
    }
