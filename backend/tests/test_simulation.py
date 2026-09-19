"""SupplyChain Sentinel - Stage 5: what-if simulator tests (pytest).

Covers: each of the 4 demo scenarios end-to-end, the no-mutation guarantee
(simulation must work on a copy), before/after bookkeeping, error handling,
and the API contract.

Run from the project root:  .venv/Scripts/python -m pytest backend/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.graph import EDGES, NODES, build_graph, to_react_flow  # noqa: E402
from backend.propagation import DECAY, MAX_HOPS  # noqa: E402
from backend.simulation import SCENARIOS, run_simulation  # noqa: E402
from backend.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


URL = "/api/simulation"


def graph_snapshot() -> dict[str, float]:
    """Baseline risks of the REAL network (not a copy)."""
    g = build_graph()
    return {n: float(d.get("risk", 0.0)) for n, d in g.nodes(data=True)}


# ------------------------------------------------------------- catalog ----
def test_four_scenarios_exist():
    assert set(SCENARIOS) >= {
        "supplier_failure", "port_closure", "transportation_disruption", "demand_spike"
    }


def test_scenario_catalog_endpoint(client):
    res = client.get(f"{URL}/scenarios")
    assert res.status_code == 200
    data = res.json()
    ids = {s["scenario"] for s in data["scenarios"]}
    assert {"supplier_failure", "port_closure", "transportation_disruption", "demand_spike"} <= ids


# ------------------------------------------------- scenario 1: supplier ----
def test_scenario_supplier_failure():
    res = run_simulation("supplier_failure")  # default target sup_c
    assert res["scenario"] == "supplier_failure"
    assert res["target_node"] == "sup_c"
    assert res["event_risk"] == 0.95

    # before/after bookkeeping
    assert res["before_risk"]["sup_c"] == pytest.approx(0.30)  # static baseline
    assert res["after_risk"]["sup_c"] == pytest.approx(0.95)   # event applied
    ids = {a["node_id"] for a in res["affected_nodes"]}
    assert {"fac_2", "route_rail", "wh_2", "wh_3", "cus_2", "cus_3"} <= ids

    # formula check on the top node: 0.95 * weight(sup_c->fac_2)=0.85 * DECAY
    top = res["affected_nodes"][0]
    assert top["node_id"] == "fac_2"
    assert top["after_risk"] == pytest.approx(round(0.95 * 0.85 * DECAY, 4), abs=1e-9)
    assert top["delta"] == pytest.approx(
        round(0.95 * 0.85 * DECAY, 4) - res["before_risk"]["fac_2"], abs=1e-9
    )
    # source node itself must not appear as affected (it IS the event)
    assert "sup_c" not in ids
    # supplier-switch rule available: alternates for sup_c's factory exist
    assert "R1" in res["summary"]["rules_fired"] or any(
        r["rule_id"] == "R1" for r in res["recommendations"]
    ) or res["recommendation_context"]["supplier_risk"] <= 70  # or honestly below threshold


# --------------------------------------------------- scenario 2: port ----
def test_scenario_port_closure():
    res = run_simulation("port_closure")
    assert res["target_node"] == "port_lb"
    assert res["after_risk"]["port_lb"] == pytest.approx(0.95)
    ids = {a["node_id"] for a in res["affected_nodes"]}
    assert {"wh_1", "wh_2", "cus_1", "cus_2"} <= ids
    # customers affected through warehouse hops only (2 hops)
    assert all(a["hops"] <= MAX_HOPS for a in res["affected_nodes"])
    # reroute/monitor style responses expected; transport risk override 92 > 70
    assert res["recommendation_context"]["transport_risk"] > 70
    assert "R3" in res["summary"]["rules_fired"]


# ------------------------------------------- scenario 3: transportation ----
def test_scenario_transportation_disruption():
    res = run_simulation("transportation_disruption")
    assert res["target_node"] == "route_rail"
    assert res["after_risk"]["route_rail"] == pytest.approx(0.90)
    ids = {a["node_id"] for a in res["affected_nodes"]}
    assert {"wh_2", "wh_3"} <= ids
    # lead-time override (27 > 21) should trigger R5
    assert "R5" in res["summary"]["rules_fired"]
    # delta ordering: warehouses (1 hop) above customers (2+ hops)
    hops = {a["node_id"]: a["hops"] for a in res["affected_nodes"]}
    deltas = {a["node_id"]: a["delta"] for a in res["affected_nodes"]}
    for wh in ("wh_2", "wh_3"):
        for cus in ("cus_2", "cus_3"):
            if wh in hops and cus in hops:
                assert deltas[wh] > deltas[cus]


# ------------------------------------------------ scenario 4: demand ----
def test_scenario_demand_spike():
    res = run_simulation("demand_spike")
    assert res["target_node"] == "wh_2"
    assert res["after_risk"]["wh_2"] == pytest.approx(0.55)
    # demand override must land in the recommendation context
    ctx = res["recommendation_context"]
    assert ctx["demand"] == 18000.0
    assert ctx["inventory_ratio"] == pytest.approx(0.3)
    # thin stock + huge demand -> expedite rule
    assert "R4" in res["summary"]["rules_fired"]
    # milder event: fewer/less-severe effects than supplier_failure
    sup = run_simulation("supplier_failure")
    assert res["summary"]["max_delta"] < sup["summary"]["max_delta"]


# --------------------------------------------------- copy semantics ----
def test_real_graph_is_never_mutated():
    snapshot = graph_snapshot()
    edge_snapshot = [(u, v, d["weight"]) for u, v, d in build_graph().edges(data=True)]

    # run every scenario aggressively, with custom targets and risks
    run_simulation("supplier_failure", target_node="sup_a", event_risk=1.0)
    run_simulation("port_closure", event_risk=1.0)
    run_simulation("transportation_disruption", event_risk=1.0)
    run_simulation("demand_spike", target_node="wh_1", event_risk=1.0, demand_multiplier=3.0)

    after = graph_snapshot()
    assert after == snapshot, "real graph node risks must be untouched"
    assert [(u, v, d["weight"]) for u, v, d in build_graph().edges(data=True)] == edge_snapshot
    # react-flow serializer still reflects the pristine network
    assert to_react_flow()["node_count"] == len(NODES)


def test_custom_target_and_overrides():
    res = run_simulation("supplier_failure", target_node="sup_a", event_risk=0.8)
    assert res["target_node"] == "sup_a"
    assert res["after_risk"]["sup_a"] == pytest.approx(0.80)
    assert res["affected_nodes"][0]["node_id"] == "fac_1"
    assert res["affected_nodes"][0]["after_risk"] == pytest.approx(
        round(0.8 * 0.80 * DECAY, 4), abs=1e-9
    )


# ------------------------------------------------------ error handling ----
def test_unknown_scenario_returns_404(client):
    res = client.post(URL, json={"scenario": "zombie_apocalypse"})
    assert res.status_code == 404


def test_unknown_target_node_returns_404(client):
    res = client.post(URL, json={"scenario": "supplier_failure", "target_node": "supplier_3"})
    assert res.status_code == 404
    assert "supplier_3" in res.json()["detail"]


def test_wrong_node_type_returns_422(client):
    res = client.post(URL, json={"scenario": "port_closure", "target_node": "sup_a"})
    assert res.status_code == 422
    assert "port" in res.json()["detail"]


def test_invalid_event_risk_returns_422(client):
    res = client.post(URL, json={"scenario": "supplier_failure", "event_risk": 1.5})
    assert res.status_code == 422
    res = client.post(URL, json={"scenario": "supplier_failure", "event_risk": 0.0})
    assert res.status_code == 422


def test_invalid_demand_multiplier_returns_422(client):
    res = client.post(URL, json={"scenario": "demand_spike", "demand_multiplier": -2})
    assert res.status_code == 422


# ------------------------------------------------------- API contract ----
def test_response_contract(client):
    res = client.post(URL, json={"scenario": "supplier_failure"})
    assert res.status_code == 200
    data = res.json()
    for key in ("scenario", "affected_nodes", "before_risk", "after_risk",
                "propagation_paths", "recommendations"):
        assert key in data, f"missing required key: {key}"
    assert data["summary"]["graph_modified"] is False
    # every affected node has before/after/delta and after >= before
    for a in data["affected_nodes"]:
        assert {"node_id", "before_risk", "after_risk", "delta", "hops"} <= set(a.keys())
        assert a["after_risk"] >= a["before_risk"]
        assert a["delta"] == pytest.approx(a["after_risk"] - a["before_risk"], abs=1e-6)
    # propagation paths include the dominant path + edge detail
    p = data["propagation_paths"][0]
    assert len(p["dominant_path"]) >= 2
    assert all({"from", "to", "relationship", "weight"} <= set(e.keys()) for e in p["path_edges"])
    # recommendations carry the Stage 4 contract
    for r in data["recommendations"]:
        assert {"action", "reason", "risk_reduction", "priority", "cost_estimate"} <= set(r.keys())
